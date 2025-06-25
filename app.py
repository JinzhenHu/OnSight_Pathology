import sys
import math
import time
import json
import cv2
import numpy as np
import mss
import torch
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox,
    QLineEdit, QGroupBox, QHBoxLayout, QFrame, QMessageBox, QStyledItemDelegate,
    QSizePolicy, QGridLayout, QDialog, QTextEdit, QFileDialog, QProgressDialog,QInputDialog
)
from PyQt6.QtGui import QPixmap, QImage, QStandardItem, QStandardItemModel

from PyQt6.QtGui import QShortcut, QKeySequence,QIcon

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRect, QSize, QTimer
from pynput import mouse
from PIL import Image
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from utils import load_model, resource_path
from llm_manager import load_llm   
from llm_manager import stream_reply
from dataclasses import dataclass
print("Torch version:", torch.__version__)


##############################################################################################################################################
# VLM Model List
##############################################################################################################################################

LLM_CATALOG = {    
    "Internvl3-2b(new)":    "metadata/llm_internvl3_2b.json",   
    "Internvl3-8b(new)":    "metadata/llm_internvl3_8b.json",
    "Internvl2-2b(old)":    "metadata/llm_internvl2_2b.json", 
    "Internvl2-8b(old)":    "metadata/llm_internvl2_8b.json",    
    "Qwen-VL-Chat":    "metadata/llm_qwen.json",            
   # "Bio-Medical LLaMA-3":  "metadata/llm_biomed_llama3.json",
}

##############################################################################################################################################
# Dropdown model metadata
##############################################################################################################################################

dropdown_categories = [
    ("▶️ Classification Models", [
         #{'name': "Tumor Compact (VGG19)", 'info_file': 'metadata/tumor_compact_vgg.json'},
         #{'name': "Tumor Compact (EfficientNetV2) (Test)", 'info_file': 'metadata/tumor_compact_efficientnet.json'},
        {'name': "Prior 16-class Tumor Compact (VIT)", 'info_file': 'metadata/tumor_compact_vit.json'},
        {'name': "New Tumor 4-Class (VIT)", 'info_file': 'metadata/tumor_compact_kaiko_vit.json'},
       {'name': "New Tumor 4-Class (Resnet)", 'info_file': 'metadata/tumor_compact_resnet.json'},
       #{'name': "GliomaAstroOligo(VIT)", 'info_file': 'metadata/glio_vit.json'}
    ]),
    ("▶️ Segmentation Models", [
        {'name': "MIB (YOLO, Segmentation)", 'info_file': 'metadata/mib_yolo_1024.json'},
         {'name': "MIB (YOLO, Bounding Boxes)", 'info_file': 'metadata/mib_yolo_1024_det.json'},
        {'name': "MIB (Mask R-CNN)", 'info_file': 'metadata/mib_mrcnn.json'}
    ]),
    ("▶️ Object Detection Models", [
        {'name': "Mitosis Detection (Retinanet)", 'info_file': 'metadata/mib_mitosis.json'}
    ]),
]

model_to_info = {}
for _, _v in dropdown_categories:
    for __v in _v:
        with open(resource_path(__v['info_file'])) as f:
            model_to_info[__v['name']] = json.load(f)

###############################################################################################################################################
#Data Class for aggregator function
###############################################################################################################################################
@dataclass
class AggregateStats:
    total_area_mm2: float = 0.0
    conf_sum: float       = 0.0
    n_tiles: int          = 0
    mitosis: int          = 0
    mib_pos: int          = 0        
    mib_total: int        = 0      

    def reset(self):
        self.total_area_mm2 = self.conf_sum = self.mitosis = 0
        self.n_tiles = self.mib_pos = self.mib_total = 0


##############################################################################################################################################
#LLM Worker
##############################################################################################################################################
class _LLMWorker(QThread):
    finished = pyqtSignal(str)        

    def __init__(self, model, tokenizer,llm_cfg, pil_img, msgs, prompt):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.llm_cfg = llm_cfg 
        self.pil_img = pil_img
        self.msgs = msgs
        self.prompt = prompt


    def run(self):
        reply, self.msgs = stream_reply(
            self.model, self.tokenizer,self.llm_cfg,
            self.pil_img, self.prompt,
            self.msgs)
        self.finished.emit(reply)

##############################################################################################################################################
#LLM chat Dialog
##############################################################################################################################################
class LLMChatDialog(QDialog):
    """One-off chat about the latest inference frame."""

    def __init__(self, frame_rgb: np.ndarray, parent=None):
        super().__init__(parent)

        # ----------------------- GUI  -----------------------
        self.setWindowTitle("Chat with LLM")
        self.resize(480, 600)


        pil_img = Image.fromarray(frame_rgb)     
        self.pil_img = pil_img

        cfg_path = LLM_CATALOG[parent.cmb_llm.currentText()]
        self.model, self.tokenizer, self.llm_cfg = load_llm(cfg_path)
        self.msgs = []  # chat history


        # ---------------------- GUI  -----------------------
        vbox = QVBoxLayout(self)

        lbl_img = QLabel()                    
        lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        h, w, c = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, c*w, QImage.Format.Format_RGB888)
        lbl_img.setPixmap(QPixmap.fromImage(qimg).scaled(
            300, 300, Qt.AspectRatioMode.KeepAspectRatio))
        vbox.addWidget(lbl_img)

        self.txt_history = QTextEdit()         
        self.txt_history.setReadOnly(True)
        vbox.addWidget(self.txt_history)

        self.inp = QLineEdit()                   
        self.inp.setPlaceholderText("Ask something about the image …")
        vbox.addWidget(self.inp)

        btn_send = QPushButton("Send")      
        btn_send.clicked.connect(self.on_send)
        vbox.addWidget(btn_send)


    # ---------------------Send the Prompt to LLM------------------------
    def on_send(self):
        text = self.inp.text().strip()
        if not text:
            return
        self.inp.clear()

        # 1) show the user’s question 
        self._append("User", text)

        # 2) start background thread
        self.worker = _LLMWorker(self.model, self.tokenizer,self.llm_cfg,
                                self.pil_img, self.msgs, text)
        self.worker.finished.connect(self._on_reply)
        self.worker.start()

        # disable the input 
        self.inp.setEnabled(False)


    # -----------------------Recieve the Response from LLM-----------------------------------
    def _on_reply(self, reply_text):
        
        self._append("Assistant", reply_text)
        self.inp.setEnabled(True)
        self.inp.setFocus()

    def _append(self, who, txt):
        self.txt_history.append(f"<b>{who}:</b> {txt}")

##############################################################################################################################################
# Worker thread 
##############################################################################################################################################

class ClassificationThread(QThread):
    #update_image = pyqtSignal(np.ndarray, str)  
    update_image = pyqtSignal(np.ndarray, str, dict)

    def __init__(self, ui_instance, model_name):
        super().__init__()
        self.ui = ui_instance 
        self.running = True
        self.paused = False

        res = load_model(model_to_info[model_name])
        self.model = res["model"]
        self.process_region = res["process_region_func"]
        self.using_gpu = res['using_gpu']
        self.metadata = model_to_info[model_name]

    def run(self):
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue
            t0 = time.time()
            extra_cfg = {
                k: w.text() for k, w in self.ui.additional_config_inputs.items()
            }
            frame, res_txt, metrics  = self.process_region(
                self.ui.selected_region,
                model=self.model,
                metadata=self.metadata,
                additional_configs=extra_cfg,
            )
            res_txt += f"\n({time.time()-t0:.2f}s)"
            # self.update_image.emit(frame, res_txt)
#######################################################################
#Aggregate Function update image new version
#######################################################################
            self.update_image.emit(frame, res_txt, metrics)
#######################################################################
#######################################################################
            time.sleep(0.1)

    def stop(self):
        self.running = False
        self.wait()             

        # -------- free the model from GPU memory ---------
        if self.using_gpu:
            del self.model
            torch.cuda.empty_cache()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

##############################################################################################################################################
# Main GUI 
##############################################################################################################################################

class ImageClassificationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Realtime Inference")

        # Public state
        self.selected_region = None
        self.additional_config_inputs = {}
        self.thread = None
        self.latest_frame = None  # np.ndarray RGB
        self.last_result = ""

        self._build_ui()
##############################################################################################################################################
#Aggregate Function new functions
##############################################################################################################################################
        self.agg = AggregateStats()      
        self.latest_metrics = None  
        self.agg_active = False
        self.lbl_help.setVisible(False) 
        self.inference_ready = False
        self.agg_records: list[dict] = [] 
        self.setWindowIcon(QIcon(resource_path("sample_icon")))

    def _agg_start(self):
        if not self.inference_ready:
            QMessageBox.information(
                self, "Not ready",
                "Run inference first, then press Start to begin aggregating.")
        self.agg_active = True
        self.lbl_agg.setText(self._agg_text() or "Collecting …")
        self.btn_agg_start.setEnabled(False)
        self.btn_agg_stop.setEnabled(True)
        self.lbl_help.setVisible(True)
        self.btn_reset.setEnabled(True) 
        self.btn_agg_stop.setFocus() 

    def _agg_stop(self):
        self.agg_active = False
        self.btn_agg_start.setEnabled(True)
        self.btn_agg_stop.setEnabled(False)
        self.lbl_help.setVisible(False)

    def _on_reset(self):
        self.agg.reset()
        self.lbl_agg.setText("Analysis pending...")
        self.agg_active = False
        self.btn_agg_start.setEnabled(True)
        self.btn_agg_stop.setEnabled(False)
        self.btn_reset.setEnabled(False)      # disable until next Start
        self.btn_agg_stop.setFocus() 
        self.agg_records.clear()
        self.btn_agg_export.setEnabled(False)

    def _agg_export(self):
        if not self.agg_records:
            QMessageBox.information(self, "Nothing to export", "No tiles added yet.")
            return

        # choose folder
        out_dir = QFileDialog.getExistingDirectory(
            self, "Select export folder", "")
        if not out_dir:
            return

        # save images & gather rows
        rows = []
        for idx, rec in enumerate(self.agg_records, start=1):
            base = f"{out_dir}/{idx}"
            orig_path  = base + ".jpg"
            anno_path  = base + "_annotated.jpg"

            cv2.imwrite(orig_path,  cv2.cvtColor(rec["orig"],  cv2.COLOR_RGB2BGR))
            cv2.imwrite(anno_path,  cv2.cvtColor(rec["annot"], cv2.COLOR_RGB2BGR))

            row = rec["metrics"].copy()
            row["file_orig"]      = orig_path
            row["file_annotated"] = anno_path
            rows.append(row)

        # write CSV 
        csv_path = f"{out_dir}/aggregate_metrics.csv"
        import csv
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        QMessageBox.information(
            self, "Export complete",
            f"Saved {len(rows)} images and metrics to:\n{out_dir}")

    # ======================================================================== 
    # Aggregate helpers 
    # ========================================================================
    
    def _on_add(self):
        if not self.agg_active or not self.latest_metrics:
            return
        meta = model_to_info[self.cmb_model.currentText()]
        agg_type = meta.get("aggregate_type", "classification")
        m  = self.latest_metrics
        mm2 = m["area_px"] * (m["mpp"]/1000)**2

        # ---------- Mitosis Task----------
        if agg_type == "mitosis":
            count, ok = QInputDialog.getInt(
                self, "Confirm mitosis count",
                "Detected mitoses (edit if needed):",
                m["mitosis"], 0, 999)
            if not ok:
                return
            add_conf   = 0
            add_pos = add_tot = 0
            add_mitos  = count
            new_metrics = {
                "Mitosis number": count,
                "Tissue area": mm2,
                "Mpp": m["mpp"]
            }
        # ---------- Mib Task -------------
        elif agg_type == "mib":
            pct = (m["mib_pos"] / m["mib_total"] * 100
                if m["mib_total"] else 0)
            reply = QMessageBox.question(
                self, "Add this tile?",
                (f"(+) cells: {m['mib_pos']}\n"
                f"(-) cells: {m['mib_total']-m['mib_pos']}\n"
                f"Ki-67%: {pct:.1f}\n"
                f"Tile area: {mm2:.3f} mm²\n\n"
                "Add to aggregate?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return
            add_conf   = 0
            add_mitos  = 0
            add_pos    = m["mib_pos"]
            add_tot    = m["mib_total"]
            new_metrics = {
                "Ki-67%": pct,
                "Positive cell number": m['mib_pos'],
                "Total cell number": m['mib_total'],
                "Tissue area": mm2,
                "Mpp": m["mpp"]
            }
        # ---------- Classification Task ----------
        elif agg_type == "classification":
            reply = QMessageBox.question(
                self, "Add this tile?",
                (f"Confidence = {m['conf']:.3f}\n"
                f"Tile area   = {mm2:.3f} mm²\n\n"
                "Add to aggregate?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            add_conf  = m["conf"]
            add_mitos = 0
            add_pos = add_tot = 0
            new_metrics = {
                "Confidence score": m["conf"],
                "Tissue area": mm2,
                "Mpp": m["mpp"]
            }

        # ---------- Accumulate Function ----------
        self.agg_records.append({
            "annot": self.latest_frame.copy(),
            "orig":  self.latest_original.copy(),
            "metrics": new_metrics,               
        })
        self.btn_agg_export.setEnabled(True) 
        self.agg.total_area_mm2 += mm2 
        self.agg.conf_sum       += add_conf
        self.agg.n_tiles        += 1
        self.agg.mitosis        += add_mitos
        self.agg.mib_pos        += add_pos
        self.agg.mib_total      += add_tot
        self.lbl_agg.setText(self._agg_text())


    def _agg_text(self):
        if self.agg.n_tiles == 0:
            return "Analysis pending..."

        meta      = model_to_info[self.cmb_model.currentText()]
        agg_type  = meta.get("aggregate_type", "classification")

        lines = [
            f"Total Tiles: {self.agg.n_tiles}",
            f"Total Area:  {self.agg.total_area_mm2:.3f} mm²"
        ]

        if agg_type == "classification":
            avg = self.agg.conf_sum / self.agg.n_tiles if self.agg.n_tiles else 0
            lines.append(f"Average confidence: {avg:.3f}")

        elif agg_type == "mitosis":
            density = (self.agg.mitosis / self.agg.total_area_mm2
                    if self.agg.total_area_mm2 else 0)
            lines.append(f"Mitoses: {self.agg.mitosis}  "
                        f"({density:.2f} / mm²)")
        elif agg_type == "mib":
            pct = (self.agg.mib_pos / self.agg.mib_total * 100
                if self.agg.mib_total else 0)
            lines += [
                f"(+) cells: {self.agg.mib_pos}",
                f"(-) cells: {self.agg.mib_total - self.agg.mib_pos}",
                f"Average Ki-67%: {pct:.2f}"
            ]

        return "\n".join(lines)

################################################################################################################################################
################################################################################################################################################
    # --------------------- UI --------------------------------- 
    def _build_ui(self):
        
        main_left = QVBoxLayout()  
        main_right = QVBoxLayout()  
        # -------------------- Model selection ------------------------------
        grp_model = QGroupBox("Model Selection")
        h_model = QHBoxLayout()

        class PaddedItemDelegate(QStyledItemDelegate):
            PADDING = 8
            def paint(self, painter, option, index):
                option.rect = QRect(
                    option.rect.x() + self.PADDING,
                    option.rect.y() + self.PADDING // 2,
                    option.rect.width() - 2 * self.PADDING,
                    option.rect.height() - self.PADDING,
                )
                super().paint(painter, option, index)
            def sizeHint(self, option, index):
                s = super().sizeHint(option, index)
                return QSize(s.width(), s.height() + self.PADDING)

        self.cmb_model = QComboBox()
        self.cmb_model.setItemDelegate(PaddedItemDelegate(self.cmb_model))
        mdl = QStandardItemModel()
        for cat, items in dropdown_categories:
            header = QStandardItem(cat)
            f = header.font(); f.setBold(True)
            header.setFont(f)
            header.setEnabled(False)
            mdl.appendRow(header)
            for it in items:
                mdl.appendRow(QStandardItem(it["name"]))
        self.cmb_model.setModel(mdl)
        self.cmb_model.currentIndexChanged.connect(self._on_model_changed)
        h_model.addWidget(self.cmb_model)



        # Info button
        btn_info = QPushButton("ℹ️")
        btn_info.setFixedSize(30, 30)
        btn_info.clicked.connect(self._show_model_info)
        h_model.addWidget(btn_info)

        grp_model.setLayout(h_model)
        main_left.addWidget(grp_model)
        # -------------------- LLM selection -----------------------------
        grp_llm = QGroupBox("GPT Selection")
        h_llm   = QHBoxLayout()
        self.cmb_llm = QComboBox()
        self.cmb_llm.addItems(LLM_CATALOG.keys())
        h_llm.addWidget(self.cmb_llm)
        grp_llm.setLayout(h_llm)
        main_left.addWidget(grp_llm)
        # -------------------- Additional configs --------------------------
        self.grp_cfg = QGroupBox("Additional Configs")
        self.v_cfg = QVBoxLayout()
        self.grp_cfg.setLayout(self.v_cfg)
        main_left.addWidget(self.grp_cfg)

        # -------------------- Screen capture ------------------------------
        grp_cap = QGroupBox("Screen Capture")
        v_cap = QVBoxLayout()

        self.lbl_cap_hint = QLabel("Recommended Capture Size:")
        self.lbl_cap_hint.setStyleSheet("color:red;")
        v_cap.addWidget(self.lbl_cap_hint)

        btn_sel = QPushButton("Select Screen Region")
        btn_sel.clicked.connect(self._select_region)
        v_cap.addWidget(btn_sel)

        self.lbl_region = QLabel("Selected Region: NOT SET")
        self.lbl_region.setStyleSheet("color:grey;")
        v_cap.addWidget(self.lbl_region)

        grp_cap.setLayout(v_cap)
        main_left.addWidget(grp_cap)

        main_left.addStretch()

        # ----------- Controls (start/stop/export/chat) ----------
        grp_ctrl = QGroupBox("Inference Controls")
        v_ctrl = QVBoxLayout()

        ##### USING GPU STATUS ICON
        status_widget = QWidget()
        status_layout = QHBoxLayout(status_widget)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(6)  

        from custom_widgets.PulsingDot import PulsingDot
        self.using_gpu_icon = PulsingDot(color="grey")
        label = QLabel("Using GPU")
        label.setStyleSheet("QLabel { margin-top: -3px;  }")  

        status_layout.addWidget(self.using_gpu_icon)
        status_layout.addWidget(label)

        v_ctrl.addWidget(status_widget)

        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self._start)
        self.btn_start.setEnabled(False)
        v_ctrl.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self._stop)
        self.btn_stop.setEnabled(False)
        v_ctrl.addWidget(self.btn_stop)

        # Export button ---------------------------------------------------
        self.btn_export = QPushButton("Export")
        self.btn_export.clicked.connect(self._export)
        self.btn_export.setEnabled(False)
        v_ctrl.addWidget(self.btn_export)

        # Chat button -----------------------------------------------------
        self.btn_chat = QPushButton("Chat with LLM")
        self.btn_chat.clicked.connect(self._open_chat)
        self.btn_chat.setEnabled(False)
        v_ctrl.addWidget(self.btn_chat)

        grp_ctrl.setLayout(v_ctrl)
        main_right.addWidget(grp_ctrl)
        ##############################################################################################################################################
        #Aggregate Function UI
        ##############################################################################################################################################
        # -------------------- Aggregate Scorer --------------------
        grp_agg = QGroupBox("Aggregate Function")
        v_agg   = QVBoxLayout()
        h_agg_btns = QHBoxLayout()

        self.btn_agg_start = QPushButton("Start")
        self.btn_agg_start.clicked.connect(self._agg_start)
        h_agg_btns.addWidget(self.btn_agg_start)

        self.btn_agg_stop = QPushButton("Stop")
        self.btn_agg_stop.clicked.connect(self._agg_stop)
        self.btn_agg_stop.setEnabled(False)       
        h_agg_btns.addWidget(self.btn_agg_stop)

        h_agg_btns.addStretch()
        v_agg   = QVBoxLayout()

        v_agg.addLayout(h_agg_btns)      
        self.btn_agg_export = QPushButton("Export Aggregate")
        self.btn_agg_export.setEnabled(False)        
        self.btn_agg_export.clicked.connect(self._agg_export)
        v_agg.addWidget(self.btn_agg_export)


        self.lbl_agg = QLabel("Analysis pending...")
        v_agg.addWidget(self.lbl_agg)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self._on_reset)
        v_agg.addWidget(self.btn_reset)
        self.lbl_help = QLabel(
            "Tip ▶ Press <kbd>Spacebar</kbd> to add this view to the aggregate."
        )
        self.lbl_help.setStyleSheet(
            "QLabel { color: grey; font-style: italic; margin-top: 4px; }"
        )
        v_agg.addWidget(self.lbl_help)

        grp_agg.setLayout(v_agg)
        main_right.addWidget(grp_agg)
        sc_add = QShortcut(QKeySequence("Space"), self)  
        sc_add.setContext(Qt.ShortcutContext.ApplicationShortcut)
        sc_add.activated.connect(self._on_add)
        self.shortcut_add = sc_add          
        self.btn_agg_start.setEnabled(False) 
        self.btn_reset.setEnabled(False)

        ################################################################################################################################################
        ################################################################################################################################################
        # -------------------- Display output -----------------------------
        grp_out = QGroupBox("Inference Output")
        v_out = QVBoxLayout()

        self.lbl_img = QLabel()
        self.lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v_out.addWidget(self.lbl_img)

        self.lbl_res = QLabel("Result:")
        v_out.addWidget(self.lbl_res)

        grp_out.setLayout(v_out)
        main_right.addWidget(grp_out)

        main_right.addStretch()

        # -------------------- Assemble main layout -----------------------
        w_left = QWidget(); w_left.setLayout(main_left)
        w_right = QWidget(); w_right.setLayout(main_right)
        w_left.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        w_right.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        h_main = QHBoxLayout(self)
        h_main.addSpacing(10)
        h_main.addWidget(w_left, alignment=Qt.AlignmentFlag.AlignTop)
        h_main.addWidget(w_right, alignment=Qt.AlignmentFlag.AlignTop)

        self._on_model_changed() 


    # ------------- UX ---------------
    def _select_region(self):
        def get_mouse_pos():
            pos = []
            def _click(x, y, button, pressed):
                if pressed:
                    pos.append((x, y)); return False
            with mouse.Listener(on_click=_click) as l: l.join()
            return pos[0] if pos else (0, 0)

        x1, y1 = get_mouse_pos(); x2, y2 = get_mouse_pos()
        self.selected_region = {
            "left": min(x1, x2), "top": min(y1, y2),
            "width": abs(x1-x2), "height": abs(y1-y2),
        }
        self.lbl_region.setText(
            f"Selected Region:\nWidth: {self.selected_region['width']:.0f} px\n"
            f"Height: {self.selected_region['height']:.0f} px")
        self.lbl_region.setStyleSheet("color:green;")
        self.btn_start.setEnabled(True)

    # ------------------------------------- Thread------------------------------------------------------
    def _start(self):
        if not hasattr(self, '_disclaimer_shown'):
            #Disclaimer
            QMessageBox.warning(
                self, "Disclaimer",
                "This tool is intended for research and educational purposes only. "
                "It is not a medical tool and should not be used for clinical diagnosis, "
                "patient management, or any other medical purpose. The accuracy of the "
                "models has not been validated for clinical use."
            )
            self._disclaimer_shown = True

        if self.selected_region is None:
            QMessageBox.warning(self, "No region", "Please select a screen region first.")
            return
        if self.thread and self.thread.isRunning():
            self.thread.stop()       
            self.thread.wait()        
            self.thread = None

        model_name = self.cmb_model.currentText()
        self.thread = ClassificationThread(self, model_name)
        self.thread.update_image.connect(self._update_display)
        self.thread.start()
        if self.thread.using_gpu:
            self.using_gpu_icon.set_color("green")
        else:
            self.using_gpu_icon.set_color("red")

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_chat.setEnabled(True)

    def _stop(self):
        if self.thread:
            self.thread.stop(); self.thread.wait(); self.thread = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_chat.setEnabled(False)
        self.using_gpu_icon.set_color("grey")

    # ---------------------------------- Display----------------------------------
    def _update_display(self, frame, txt, metrics):
        self.latest_metrics = metrics   
        self.latest_frame    = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB).copy()   # annotated
        self.latest_original = metrics.get("orig_img", self.latest_frame).copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        h, w, c = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, c*w, QImage.Format.Format_RGB888)
        self.lbl_img.setPixmap(QPixmap.fromImage(qimg).scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))
        self.lbl_res.setText(f"Result: {txt}")

        self.latest_frame = frame_rgb.copy()
        self.last_result = txt
        self.btn_export.setEnabled(True)
        if not self.inference_ready:
            self.inference_ready = True
            self.btn_agg_start.setEnabled(True)
    # --------------------------------- Export------------------------------------------
    def _export(self):
        if self.latest_frame is None:
            QMessageBox.warning(self, "No data", "Nothing to export yet!")
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Export Image", "", "PNG Image (*.png)")
        if not save_path:
            return
        cv2.imwrite(save_path, cv2.cvtColor(self.latest_frame, cv2.COLOR_RGB2BGR))
        with open(save_path.rsplit(".", 1)[0] + ".txt", "w", encoding="utf-8") as fh:
            fh.write("Model: " + self.cmb_model.currentText() + "\n")
            fh.write("Result: " + self.last_result + "\n")
        QMessageBox.information(self, "Export", f"Saved to {save_path}")

    # --------------------------------- Chat-------------------------------------------------
    def _open_chat(self):
        if self.latest_frame is None:
            return

        # ------------------------- Reminder for LLM-----------------------------
        reply = QMessageBox.question(
            self, "Load LLM?",
            "Loading the LLM for the first time can take a long time.\n"
            "Do you want to continue? (This message can be ignored on subsequent loads.)?\n"
            "Note: Please load only one model at a time to avoid running out of disk space",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.No:
            return
        if self.thread:
           # self.thread.pause()
            self.thread.stop()     # frees GPU memory
            self.thread = None
        dlg = LLMChatDialog(self.latest_frame, self)
        dlg.exec()
        if self.thread:
            self.thread.resume()
    # ------------------------ Helpers-----------------------------------------
    def _show_model_info(self):
        info = model_to_info[self.cmb_model.currentText()].get("info", "No info.")
        QMessageBox.information(self, "Model Info", info)

    def _on_model_changed(self):
        """
        Stop the current classification thread before switching models.
        """
        if self.thread and self.thread.isRunning():
            self._stop()                       # kill the old worker
        name = self.cmb_model.currentText()
        
        name = self.cmb_model.currentText()
        meta = model_to_info[name]

        tile = meta["tile_size"]
        mpp = meta.get("mpp", 0.25)
        mag = math.ceil(0.25 / mpp * 40)
        self.lbl_cap_hint.setText(
            "Recommended Capture Size:\n" +
            f"This model was trained on {tile}×{tile}px tiles at {mag}×.\n" +
            "Smaller regions will be upscaled to match the training tile size; larger ones will be sliced."
        )

        # ------------------------------------Dynamic additional configs -------------------------------------
        def clear_layout(layout):
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                child_layout = item.layout()

                if widget is not None:
                    widget.deleteLater()
                elif child_layout is not None:
                    clear_layout(child_layout)  # Recursively delete children

        # Clear existing widgets
        clear_layout(self.v_cfg)

        self.additional_config_inputs.clear()
        add_cfg = meta.get("additional_configs", {})
        self.grp_cfg.setVisible(bool(add_cfg))
        for k, v in add_cfg.items():
            row = QHBoxLayout()
            lbl = QLabel(k)
            edt = QLineEdit(str(v))
            row.addWidget(lbl); row.addWidget(edt)
            self.v_cfg.addLayout(row)
            self.additional_config_inputs[k] = edt

    # ------------------------------------------Close---------------------------------------------------------
    def closeEvent(self, e):
        self._stop()
        e.accept()
def main():
    app = QApplication(sys.argv)
    win = ImageClassificationApp()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()