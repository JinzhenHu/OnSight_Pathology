import sys
import math
import time
import json
import cv2
import numpy as np
import mss
import os

from datetime import datetime
import csv
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
    # "Internvl3-2b(new)":    "metadata/llm_internvl3_2b.json",   
    #"Internvl3-8b(new)":    "metadata/llm_internvl3_8b.json",
    "Huatuo-7b":    "metadata/huatuo.json",
    # "Internvl2-2b(old)":    "metadata/llm_internvl2_2b.json", 
    # "Internvl2-8b(old)":    "metadata/llm_internvl2_8b.json",    
    # "Qwen-VL-Chat":    "metadata/llm_qwen.json",            
   # "Bio-Medical LLaMA-3":  "metadata/llm_biomed_llama3.json",
}

##############################################################################################################################################
# Dropdown model metadata
##############################################################################################################################################

dropdown_categories = [
    ("▶️ Classification Models", [
         #{'name': "Tumor Compact (VGG19)", 'info_file': 'metadata/tumor_compact_vgg.json'},
         #{'name': "Tumor Compact (EfficientNetV2) (Test)", 'info_file': 'metadata/tumor_compact_efficientnet.json'},
        #{'name': "Prior 16-class Tumor Compact (VIT)", 'info_file': 'metadata/tumor_compact_vit.json'},
        {'name': "Tumor 4-Class (VIT)", 'info_file': 'metadata/tumor_compact_kaiko_vit.json'},
       #{'name': "New Tumor 4-Class (Resnet)", 'info_file': 'metadata/tumor_compact_resnet.json'},
       #{'name': "GliomaAstroOligo(VIT)", 'info_file': 'metadata/glio_vit.json'}
    ]),
    ("▶️ Segmentation Models", [
        {'name': "MIB (YOLO)", 'info_file': 'metadata/mib_yolo.json'},
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
        self.msgs = []  


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
            res_txt += f"\n({time.time()-t0:.4f}s)"
            # print(res_txt)
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

###############################################################################################################    
# Stylesh
###############################################################################################################    
PROFESSIONAL_STYLESHEET = """
QWidget {
    background-color: #2c3e50;
    color: #ecf0f1;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 10pt;
}
QGroupBox {
    background-color: #34495e;
    border-radius: 6px;
    border: 1px solid #4a627a;
    margin-top: 20px;
    padding: 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 5px 10px;
    background-color: #4a627a;
    color: #1abc9c;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-weight: bold;
}
QPushButton {
    background-color: #3498db;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #4ea8e1;
}
QPushButton:pressed {
    background-color: #2980b9;
}
QPushButton:disabled {
    background-color: #566573;
    color: #95a5a6;
}
QLabel {
    background-color: transparent;
}
QLineEdit, QTextEdit, QComboBox {
    background-color: #2c3e50;
    border: 1px solid #566573;
    border-radius: 4px;
    padding: 6px;
    color: #ecf0f1;
}
QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
    border: 1px solid #1abc9c;
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 25px;
    border-left-width: 1px;
    border-left-color: #566573;
    border-left-style: solid;
    border-top-right-radius: 3px;
    border-bottom-right-radius: 3px;
}
QComboBox::down-arrow {
    image: url(./icons/chevron-down.svg); /* Assumes an icon file is available */
    width: 16px;
    height: 16px;
}
QComboBox QAbstractItemView {
    border: 1px solid #1abc9c;
    background-color: #34495e;
    selection-background-color: #1abc9c;
    selection-color: white;
    color: #ecf0f1;
}
QFrame {
    border: 1px solid #4a627a;
    border-radius: 4px;
    background-color: #2c3e50;
}
QMessageBox {
    background-color: #34495e;
}
QDialog {
    background-color: #2c3e50;
}
"""
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
        self.show_calib_box = False   
        self.box_mm2        = None    
        self._calib_help_shown = False

        self.cls_stats = {}   # {class: {'tiles':0, 'conf_sum':0, 'area':0}}

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

        # self.box_mm2 = None
        self.show_calib_box = False
        self.cls_stats.clear()

    # ======================================================================== 
    # Aggregate Export 
    # ========================================================================
    def _agg_export(self):
        if not self.agg_records:
            QMessageBox.information(
                self, "Nothing to export", "No tiles added yet.")
            return

        # ── Choose main folder ────────────────────────────────────────────
        parent_dir = QFileDialog.getExistingDirectory(
            self, "Select export folder", "")
        if not parent_dir:
            return

        # ── Automatic generate subfolder ─────────────────────────────────
        base_name = "Agregate_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir   = os.path.join(parent_dir, base_name)
        counter   = 1
        while os.path.exists(out_dir):         
            out_dir = os.path.join(parent_dir, f"{base_name}_({counter})")
            counter += 1
        os.makedirs(out_dir, exist_ok=True)

        # ── Save Images ───────────────────────────────────
        rows = []
        all_clsset = set()
        for idx, rec in enumerate(self.agg_records, start=1):
            base = os.path.join(out_dir, f"{idx:03d}")     # 000,001…
            orig_path = base + ".jpg"
            anno_path = base + "_annotated.jpg"

            cv2.imwrite(orig_path,
                        cv2.cvtColor(rec["orig"],  cv2.COLOR_RGB2BGR))
            cv2.imwrite(anno_path,
                        cv2.cvtColor(rec["annot"], cv2.COLOR_RGB2BGR))
            row = rec["metrics"].copy()
            if "Class" in row:                  # present only in classification
                row.pop("Confidence score", None)
            row["file_orig"]      = orig_path
            row["file_annotated"] = anno_path
            if "probs" in rec:                       # only classification
                for cls, p in rec["probs"].items():
                    row[cls] = p
                    all_clsset.add(cls)
            rows.append(row)
        
        # Skip if not classification task    
        if all_clsset:
            for r in rows:
                for cls in all_clsset:
                    r.setdefault(cls, "")

            # Add average row
            avg_row = {k: "" for k in rows[0].keys()}
            avg_row["Class"] = "AVERAGE"
            for cls in all_clsset:
                vals = [r[cls] for r in rows if r[cls] != ""]
                avg_row[cls] = sum(vals)/len(vals) if vals else ""
            rows.append(avg_row)

        # ── wrte CSV ─────────────────────────────────────────────
        csv_path = os.path.join(out_dir, "aggregate_metrics.csv")
        with open(csv_path, "w", newline="",encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        # ── Complete ───────────────────────────────────────────
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
        extra_payload = {}  

        if self.box_mm2 and not self.show_calib_box:      
            mm2 = self.box_mm2 * 9
        else:
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
                "Tissue area (mm2)": mm2,
                #"Mpp": m["mpp"]
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
                "Tissue area (mm2)": mm2,
                #"Mpp": m["mpp"]
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
                "Class": m["pred_cls"],
                "Confidence score": m["conf"],
                "Tissue area (mm2)": mm2
                #"Mpp": m["mpp"]
            }
            # ── per-class stats ─────────────────────────────────────────
            cls = m["pred_cls"]
            st  = self.cls_stats.setdefault(cls,
                {'tiles': 0, 'conf_sum': 0.0, 'area': 0.0})
            st['tiles']    += 1
            st['conf_sum'] += m["conf"]
            st['area']     += mm2
            # ────────────────────────────────────────────────────────────
            extra_payload = {"probs": m["probs"]}
        # ---------- Accumulate Function ----------
        self.agg_records.append({
            "annot": self.latest_frame.copy(),
            "orig":  self.latest_original.copy(),
            "metrics": new_metrics, 
            **extra_payload               
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
            if not self.cls_stats:                   
                avg = (self.agg.conf_sum / self.agg.n_tiles
                    if self.agg.n_tiles else 0)
                lines.append(f"Average confidence: {avg:.3f}")
            else:
                lines.append("Avg confidence per class:")
                for cls, st in self.cls_stats.items():
                    avg_c = st['conf_sum'] / st['tiles']
                    lines.append(f"  • {cls}: {avg_c:.3f} "
                                f"(n={st['tiles']})")

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
    

    # ------------------------------------------------------------
    # CALIBRATION DIALOG 
    # ------------------------------------------------------------
    def _calibrate_area(self):
        # ───────── First Click ─────────
        if not self.show_calib_box:
            # First Time Instruction Pop-up
            if not self._calib_help_shown:
                QMessageBox.information(
                    self, "How to calibrate",
                    "Step 1: A translucent box will appear.\n"
                    "Step 2: In your slide viewer draw a box that exactly overlaps it\n"
                    "    and note the physical area (mm²).\n"
                    "Step 3: Click the 'Enter Area' button and input the measured value.")
                self._calib_help_shown = True

            # Open Calibrate Box
            self.show_calib_box = True
            QApplication.processEvents()    

            # Wait for Second Click
            self.btn_calib.setText("Enter area")
            return

        # ───────── Second Click ─────────
        val, ok = QInputDialog.getDouble(
            self, "Calibrate Area",
            "Enter the measured area of the highlighted box (mm²):",
            decimals=4, min=0.0001
        )

        if ok and val > 0:
            self.box_mm2 = val
            QMessageBox.information(
                self, "Calibration stored",
                f"Calibration box = {self.box_mm2:.4f} mm²")
        else:
            QMessageBox.information(
                self, "Calibration cancelled",
                "No calibration value recorded.")

        # Close and Reset
        self.show_calib_box = False
        self.btn_calib.setText("Calibrate")


    # def _calibrate_area(self):
    #     dlg = QDialog(self)
    #     dlg.setWindowTitle("Calibrate Scale Bars")

    #     form = QGridLayout(dlg)

    #     # instruction 
    #     instr = QLabel(
    #         "Measure the length of the scale-bar in the inference window "
    #         "directly in your slide viewer and input the value here."
    #     )
    #     instr.setWordWrap(True)
    #     form.addWidget(instr, 0, 0, 1, 2)

    #     # width entry
    #     form.addWidget(QLabel("Horizontal bar (mm):"), 1, 0)
    #     edt_w = QLineEdit()
    #     edt_w.setPlaceholderText("e.g. 0.300")
    #     form.addWidget(edt_w, 1, 1)

    #     # height entry
    #     form.addWidget(QLabel("Vertical bar (mm):"), 2, 0)
    #     edt_h = QLineEdit()
    #     edt_h.setPlaceholderText("e.g. 0.300")
    #     form.addWidget(edt_h, 2, 1)

    #     # OK / Cancel
    #     btn_ok     = QPushButton("OK")
    #     btn_cancel = QPushButton("Cancel")
    #     btn_ok.clicked.connect(dlg.accept)
    #     btn_cancel.clicked.connect(dlg.reject)
    #     h_btn = QHBoxLayout()
    #     h_btn.addStretch()
    #     h_btn.addWidget(btn_ok)
    #     h_btn.addWidget(btn_cancel)
    #     form.addLayout(h_btn, 3, 0, 1, 2)


    #     if dlg.exec() != QDialog.DialogCode.Accepted:
    #         return

    #     # --- parse numbers safely ---------------------------------
    #     try:
    #         w_mm = float(edt_w.text())
    #         h_mm = float(edt_h.text() or "0")
    #     except ValueError:
    #         QMessageBox.warning(self, "Invalid input",
    #                             "Please enter numeric values.")
    #         return
    #     if w_mm <= 0:
    #         QMessageBox.warning(self, "Invalid width",
    #                             "Width must be > 0.")
    #         return

    #     self.bar_mm_w = w_mm
    #     self.bar_mm_h = h_mm if h_mm > 0 else None  

    #     msg = f"Width bar  = {self.bar_mm_w:.3f} mm"
    #     if self.bar_mm_h is not None:
    #         msg += f"\nHeight bar = {self.bar_mm_h:.3f} mm"
    #     else:
    #         msg += "\nHeight bar = (not set)"
    #     QMessageBox.information(self, "Calibration stored", msg)



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
        grid    = QGridLayout()                  

        # row 0 –– control buttons
        self.btn_agg_start = QPushButton("Start")
        self.btn_agg_start.clicked.connect(self._agg_start)
        grid.addWidget(self.btn_agg_start, 0, 0)

        self.btn_agg_stop = QPushButton("Stop")
        self.btn_agg_stop.clicked.connect(self._agg_stop)
        self.btn_agg_stop.setEnabled(False)
        grid.addWidget(self.btn_agg_stop, 0, 1)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self._on_reset)
        self.btn_reset.setEnabled(False)
        grid.addWidget(self.btn_reset, 0, 2)

        self.btn_calib = QPushButton("Calibrate")
        self.btn_calib.clicked.connect(self._calibrate_area)
        self.btn_calib.setEnabled(False)
        grid.addWidget(self.btn_calib, 0, 3)

        self.btn_agg_export = QPushButton("Export")
        self.btn_agg_export.setEnabled(False)
        self.btn_agg_export.clicked.connect(self._agg_export)
        grid.addWidget(self.btn_agg_export, 0, 4)

        # row 1 –– results + tip
        box = QFrame()
        box.setFrameShape(QFrame.Shape.Box)
        box.setLineWidth(1)
        v_box = QVBoxLayout(box)
        self.lbl_agg = QLabel("Analysis pending…")
        self.lbl_agg.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        v_box.addWidget(self.lbl_agg)
        self.lbl_help = QLabel("Tip ⌨ Press <kbd>Space</kbd> to add this tile")
        self.lbl_help.setStyleSheet("color:grey; font-style:italic;")
        v_box.addWidget(self.lbl_help)
        grid.addWidget(box, 1, 0, 1, 5)      

        grp_agg.setLayout(grid)
        main_right.addWidget(grp_agg)

        # global space-bar shortcut
        sc_add = QShortcut(QKeySequence("Space"), self)
        sc_add.setContext(Qt.ShortcutContext.ApplicationShortcut)
        sc_add.activated.connect(self._on_add)
        self.shortcut_add = sc_add

        # disable initially
        self.btn_agg_start.setEnabled(False)

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
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR).copy()
        h, w, c = frame_bgr.shape

        # CENTRAL 1/3 × 1/3  CALIBRATION BOX
        if self.show_calib_box:
            bx = int(w / 3);  by = int(h / 3)          # Box Size
            cx = (w - bx) // 2;  cy = (h - by) // 2    # Put it in the middle
            p1, p2 = (cx, cy), (cx + bx, cy + by)
            bar = min(2, max(1, h // 140))

            # Transparent
            overlay = frame_bgr.copy()
            cv2.rectangle(overlay, p1, p2, (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.18, frame_bgr, 0.82, 0, frame_bgr)

            #白描边 + 黑实线
            # cv2.rectangle(frame_bgr, (p1[0]-1, p1[1]-1), (p2[0]+1, p2[1]+1),
            #             (250, 250, 250), bar, lineType=cv2.LINE_AA)
            cv2.rectangle(frame_bgr, p1, p2, (250, 250, 250), bar, lineType=cv2.LINE_AA)


        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) 
        qimg = QImage(frame_rgb.data, w, h, c*w, QImage.Format.Format_RGB888)
        self.lbl_img.setPixmap(QPixmap.fromImage(qimg).scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))
        self.lbl_res.setText(f"Result: {txt}")

        self.latest_frame = frame_rgb.copy()
        self.last_result = txt
        self.btn_export.setEnabled(True)
        if not self.inference_ready:
            self.inference_ready = True
            self.btn_agg_start.setEnabled(True)
            self.btn_calib.setEnabled(True)
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