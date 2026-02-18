import logging
import os
import sys

# ONLY FOR CPU EXE
if os.environ.get("BUILD_TYPE", "").upper() == "CPU":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TORCH_CUDA_DUMMY_DEVICE"] = "1"

# Logging to stdout never happens in exe but ultralytics logging still tries and can crash (if not utf-8).
# Crash logging to file unaffected.
for h in logging.root.handlers[:]:
    if isinstance(h, logging.StreamHandler):
        logging.root.removeHandler(h)

if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

import crash_logging

import sys
import math
import time
import json
import cv2
import numpy as np
import gc
import os
from datetime import datetime
import csv
import torch
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox,
    QLineEdit, QGroupBox, QHBoxLayout, QFrame, QMessageBox, QStyledItemDelegate,
    QSizePolicy, QGridLayout, QDialog, QFileDialog, QInputDialog,
    QDialogButtonBox, QMainWindow, QLayout
)
from PyQt6.QtGui import QPixmap, QImage, QStandardItem, QStandardItemModel, QAction
from PyQt6.QtGui import QShortcut, QKeySequence, QIcon
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRect, QSize
from pynput import mouse
from dataclasses import dataclass

from utils import load_model, resource_path, get_gpu_memory, get_system_memory, build_precision_labels
import settings
from custom_widgets.AboutDialog import AboutDialog
from custom_widgets.CollapsibleGroupBox import CollapsibleGroupBox
from custom_widgets.DisclaimerDialog import DisclaimerDialog
from custom_widgets.LLMChatDialog import LLMChatDialog

print("Torch version:", torch.__version__)


###############################################################################################################################################
# Data Class for aggregator function
###############################################################################################################################################
@dataclass
class AggregateStats:
    total_area_mm2: float = 0.0
    conf_sum: float = 0.0
    n_tiles: int = 0
    mitosis: int = 0
    mib_pos: int = 0
    mib_total: int = 0

    def reset(self):
        self.total_area_mm2 = self.conf_sum = self.mitosis = 0
        self.n_tiles = self.mib_pos = self.mib_total = 0


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
QLineEdit, QTextEdit, QComboBox, QCheckBox {
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
QMessageBox, QDialog {
    background-color: #34495e;
}
QDialog QWidget, QDialog QLabel, QDialog QPushButton, QDialog QCheckBox {
    background-color: #34495e;
}
"""


##############################################################################################################################################
# Worker thread
##############################################################################################################################################

class ClassificationThread(QThread):
    # update_image = pyqtSignal(np.ndarray, str)
    update_image = pyqtSignal(np.ndarray, str, dict)

    def __init__(self, ui_instance, model_name):
        super().__init__()
        self.ui = ui_instance
        self.running = True

        res = load_model(settings.MODEL_METADATA[model_name])
        self.model = res["model"]
        self.process_region = res["process_region_func"]
        self.using_gpu = res['using_gpu']
        self.metadata = settings.MODEL_METADATA[model_name]

    def run(self):
        while self.running:
            t0 = time.time()
            extra_cfg = {
                k: w.text() for k, w in self.ui.additional_config_inputs.items()
            }
            frame, res_txt, metrics = self.process_region(
                self.ui.selected_region,
                model=self.model,
                metadata=self.metadata,
                additional_configs=extra_cfg,
            )
            res_txt += f"\n({time.time() - t0:.4f}s)"
            # print(res_txt)
            # self.update_image.emit(frame, res_txt)
            #######################################################################
            # Aggregate Function update image new version
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
            gc.collect()
            torch.cuda.empty_cache()


##############################################################################################################################################
# Main GUI 
##############################################################################################################################################

class ImageClassificationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OnSight")

        # Public state
        self.compact_mode = False
        self.selected_region = None
        self.additional_config_inputs = {}
        self.thread = None
        self.latest_frame = None  # np.ndarray RGB
        self.last_result = ""

        # Load persistent settings
        def get_user_settings_path(filename="settings.json"):
            local_appdata = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
            settings_dir = os.path.join(local_appdata, "OnSightPathology", "Settings")
            os.makedirs(settings_dir, exist_ok=True)
            return os.path.join(settings_dir, filename)

        self.settings_file = get_user_settings_path()
        self.settings = self._load_settings()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.central_layout = QVBoxLayout(self.central_widget)
        self.central_layout.setContentsMargins(0, 0, 0, 0)  # remove QMainWindow padding around layout
        self.central_layout.setSpacing(0)

        # Track which widget is currently active
        self.current_widget = None

        # Start with full view
        self._switch_view(compact=False)
        # Setup the menu bar
        self._create_menu_bar()

        ##############################################################################################################################################
        # Aggregate Function new functions
        ##############################################################################################################################################
        self.show_calib_box = False
        self.box_mm2 = None
        self._calib_help_shown = False

        self.cls_stats = {}  # {class: {'tiles':0, 'conf_sum':0, 'area':0}}

        self.agg = AggregateStats()
        self.latest_metrics = None
        self.agg_active = False
        self.lbl_help.setVisible(False)
        self.agg_records: list[dict] = []
        self.setWindowIcon(QIcon(resource_path("sample_icon")))

        # global space-bar shortcut. must be here since we have 2 layouts
        sc_add = QShortcut(QKeySequence("Space"), self)
        sc_add.setContext(Qt.ShortcutContext.ApplicationShortcut)
        sc_add.activated.connect(self._on_add)
        self.shortcut_add = sc_add

    def _create_menu_bar(self):
        menu_bar = self.menuBar()  # QMainWindow provides this

        # ---------------- View menu ---------------
        view_menu = menu_bar.addMenu("View")

        # Checkable action for compact view
        self.compact_action = QAction("Compact", self)
        self.compact_action.setCheckable(True)
        self.compact_action.setChecked(False)  # start in full view

        # Connect to toggle function
        self.compact_action.toggled.connect(self._toggle_view)
        view_menu.addAction(self.compact_action)

        # ---------------- Help menu ----------------
        help_menu = menu_bar.addMenu("Help")

        def show_about():
            dlg = AboutDialog(parent=self, icon_path="sample_icon.ico")
            dlg.exec()  # modal dialog

        about_action = QAction("About", self)
        about_action.triggered.connect(show_about)
        help_menu.addAction(about_action)

    def _load_settings(self):
        """Loads settings from a JSON file."""
        try:
            with open(self.settings_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}  # Return empty dict if file is missing or corrupted

    def _save_settings(self):
        """Saves current settings to the JSON file."""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def _agg_start(self):
        self.agg_active = True
        self.lbl_agg.setText(self._agg_text() or "Collecting …")
        self.btn_agg_start.setEnabled(False)
        self.btn_agg_stop.setEnabled(True)
        self.lbl_help.setVisible(True)
        self.btn_agg_stop.setFocus()

    def _agg_stop(self):
        self.agg_active = False
        self.btn_agg_start.setEnabled(True)
        self.btn_agg_stop.setEnabled(False)
        self.lbl_help.setVisible(False)

    def _on_reset(self):
        self.agg.reset()
        self.lbl_agg.setText("Analysis pending...")
        # self.agg_active = False
        # self.btn_agg_start.setEnabled(True)
        # self.btn_agg_stop.setEnabled(False)
        self.btn_reset.setEnabled(False)  # enabled after the first add
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
        out_dir = os.path.join(parent_dir, base_name)
        counter = 1
        while os.path.exists(out_dir):
            out_dir = os.path.join(parent_dir, f"{base_name}_({counter})")
            counter += 1
        os.makedirs(out_dir, exist_ok=True)

        # ── Save Images ───────────────────────────────────
        rows = []
        all_clsset = set()
        for idx, rec in enumerate(self.agg_records, start=1):
            base = os.path.join(out_dir, f"{idx:03d}")  # 000,001…
            orig_path = base + ".jpg"
            anno_path = base + "_annotated.jpg"

            cv2.imwrite(orig_path,
                        cv2.cvtColor(rec["orig"], cv2.COLOR_RGB2BGR))
            cv2.imwrite(anno_path,
                        cv2.cvtColor(rec["annot"], cv2.COLOR_RGB2BGR))
            row = rec["metrics"].copy()
            if "Class" in row:  # present only in classification
                row.pop("Confidence score", None)
            row["file_orig"] = orig_path
            row["file_annotated"] = anno_path
            if "probs" in rec:  # only classification
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
                avg_row[cls] = sum(vals) / len(vals) if vals else ""
            rows.append(avg_row)

        # ── wrte CSV ─────────────────────────────────────────────
        csv_path = os.path.join(out_dir, "aggregate_metrics.csv")
        with open(csv_path, "w", newline="", encoding='utf-8') as fh:
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

        meta = settings.MODEL_METADATA[self.cmb_model.currentText()]
        agg_type = meta.get("aggregate_type", "classification")
        m = self.latest_metrics
        extra_payload = {}

        if self.box_mm2 and not self.show_calib_box:
            mm2 = self.box_mm2 * 9
        else:
            mm2 = m["area_px"] * (m["mpp"] / 1000) ** 2

        # ---------- Mitosis Task----------
        if agg_type == "mitosis":
            count, ok = QInputDialog.getInt(
                self, "Confirm mitosis count",
                "Detected mitoses (edit if needed):",
                m["mitosis"], 0, 999)
            if not ok:
                return
            add_conf = 0
            add_pos = add_tot = 0
            add_mitos = count
            new_metrics = {
                "Mitosis number": count,
                "Tissue area (mm2)": mm2,
                # "Mpp": m["mpp"]
            }
        # ---------- Mib Task -------------
        elif agg_type == "mib":

            pct = (m["mib_pos"] / m["mib_total"] * 100
                   if m["mib_total"] else 0)
            reply = QMessageBox.question(
                self, "Add this tile?",
                (f"(+) cells: {m['mib_pos']}\n"
                 f"(-) cells: {m['mib_total'] - m['mib_pos']}\n"
                 f"Ki-67%: {pct:.1f}\n"
                 f"Tile area: {mm2:.3f} mm²\n\n"
                 "Add to aggregate?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return
            add_conf = 0
            add_mitos = 0
            add_pos = m["mib_pos"]
            add_tot = m["mib_total"]
            new_metrics = {
                "Ki-67%": pct,
                "Positive cell number": m['mib_pos'],
                "Total cell number": m['mib_total'],
                "Tissue area (mm2)": mm2,
                # "Mpp": m["mpp"]
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
            add_conf = m["conf"]
            add_mitos = 0
            add_pos = add_tot = 0
            new_metrics = {
                "Class": m["pred_cls"],
                "Confidence score": m["conf"],
                "Tissue area (mm2)": mm2
                # "Mpp": m["mpp"]
            }
            # ── per-class stats ─────────────────────────────────────────
            cls = m["pred_cls"]
            st = self.cls_stats.setdefault(cls,
                                           {'tiles': 0, 'conf_sum': 0.0, 'area': 0.0})
            st['tiles'] += 1
            st['conf_sum'] += m["conf"]
            st['area'] += mm2
            # ────────────────────────────────────────────────────────────
            extra_payload = {"probs": m["probs"]}
        # ---------- Accumulate Function ----------
        self.agg_records.append({
            "annot": self.latest_frame.copy(),
            "orig": self.latest_original.copy(),
            "metrics": new_metrics,
            **extra_payload
        })
        self.btn_agg_export.setEnabled(True)
        self.agg.total_area_mm2 += mm2
        self.agg.conf_sum += add_conf
        self.agg.n_tiles += 1
        self.agg.mitosis += add_mitos
        self.agg.mib_pos += add_pos
        self.agg.mib_total += add_tot
        self.lbl_agg.setText(self._agg_text())

        self.btn_reset.setEnabled(True)

    def _agg_text(self):
        if self.agg.n_tiles == 0:
            return "Analysis pending..."

        meta = settings.MODEL_METADATA[self.cmb_model.currentText()]
        agg_type = meta.get("aggregate_type", "classification")

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
                    lines.append(f"   • {cls}: {avg_c:.3f} "
                                 f"(n={st['tiles']})")

        elif agg_type == "mitosis":
            density = (self.agg.mitosis / self.agg.total_area_mm2
                       if self.agg.total_area_mm2 else 0)
            lines.append(f"Mitoses: {self.agg.mitosis}   "
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
    #   dlg = QDialog(self)
    #   dlg.setWindowTitle("Calibrate Scale Bars")

    #   form = QGridLayout(dlg)

    #   # instruction 
    #   instr = QLabel(
    #       "Measure the length of the scale-bar in the inference window "
    #       "directly in your slide viewer and input the value here."
    #   )
    #   instr.setWordWrap(True)
    #   form.addWidget(instr, 0, 0, 1, 2)

    #   # width entry
    #   form.addWidget(QLabel("Horizontal bar (mm):"), 1, 0)
    #   edt_w = QLineEdit()
    #   edt_w.setPlaceholderText("e.g. 0.300")
    #   form.addWidget(edt_w, 1, 1)

    #   # height entry
    #   form.addWidget(QLabel("Vertical bar (mm):"), 2, 0)
    #   edt_h = QLineEdit()
    #   edt_h.setPlaceholderText("e.g. 0.300")
    #   form.addWidget(edt_h, 2, 1)

    #   # OK / Cancel
    #   btn_ok     = QPushButton("OK")
    #   btn_cancel = QPushButton("Cancel")
    #   btn_ok.clicked.connect(dlg.accept)
    #   btn_cancel.clicked.connect(dlg.reject)
    #   h_btn = QHBoxLayout()
    #   h_btn.addStretch()
    #   h_btn.addWidget(btn_ok)
    #   h_btn.addWidget(btn_cancel)
    #   form.addLayout(h_btn, 3, 0, 1, 2)

    #   if dlg.exec() != QDialog.DialogCode.Accepted:
    #       return

    #   # --- parse numbers safely ---------------------------------
    #   try:
    #       w_mm = float(edt_w.text())
    #       h_mm = float(edt_h.text() or "0")
    #   except ValueError:
    #       QMessageBox.warning(self, "Invalid input",
    #                           "Please enter numeric values.")
    #       return
    #   if w_mm <= 0:
    #       QMessageBox.warning(self, "Invalid width",
    #                           "Width must be > 0.")
    #       return

    #   self.bar_mm_w = w_mm
    #   self.bar_mm_h = h_mm if h_mm > 0 else None  

    #   msg = f"Width bar  = {self.bar_mm_w:.3f} mm"
    #   if self.bar_mm_h is not None:
    #       msg += f"\nHeight bar = {self.bar_mm_h:.3f} mm"
    #   else:
    #       msg += "\nHeight bar = (not set)"
    #   QMessageBox.information(self, "Calibration stored", msg)
    # ------------------------------------------------------------
    # LLM option
    # ------------------------------------------------------------
    def _update_llm_options(self, cmb_llm, cmb_llm_precision, lbl_llm_precision):
        """
        Updates the precision dropdown based on the selected LLM model's metadata.
        """
        llm_name = cmb_llm.currentText()
        meta = settings.LLM_METADATA.get(llm_name, {})

        precisions = meta.get("precisions")
        cmb_llm_precision.clear()

        if precisions:

            precision_labels = build_precision_labels(get_gpu_memory(), get_system_memory())

            for tech_name, display_name in precision_labels.items():
                cmb_llm_precision.addItem(display_name, tech_name)

            # Set the default selection from metadata
            default_precision = meta.get("default_precision")
            if default_precision in precisions:
                index = cmb_llm_precision.findData(default_precision)
                if index != -1:
                    cmb_llm_precision.setCurrentIndex(index)

            # Make the dropdown and its label visible
            lbl_llm_precision.setVisible(True)
            cmb_llm_precision.setVisible(True)
        else:
            # Hide the precision selector if not applicable
            lbl_llm_precision.setVisible(False)
            cmb_llm_precision.setVisible(False)

    ###toggle settings
    def _toggle_view(self, checked):
        """
        Toggle between full and compact view.
        """
        if self.thread and self.thread.isRunning():
            self._stop()  # kill the old worker

        self.compact_mode = checked
        self._switch_view(compact=checked)

    def _switch_view(self, compact: bool):
        # Remove old widget if exists
        if self.current_widget is not None:
            self.current_widget.setParent(None)  # removes it from layout

        # Create new container for this view
        self.current_widget = QWidget()
        if compact:
            layout = QVBoxLayout(self.current_widget)
            layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
            self._build_ui_compact(layout)
        else:
            layout = QHBoxLayout(self.current_widget)
            layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
            self._build_ui(layout)

        self.central_layout.addWidget(self.current_widget)
        self.current_widget.show()

        # Resize main window to fit new layout
        self.central_widget.adjustSize()
        self.adjustSize()

    ################################################################################################################################################
    ################################################################################################################################################
    # --------------------- UI --------------------------------- 
    def _build_ui(self, main_layout):

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
        for cat, items in settings.MODEL_CATELOG:
            header = QStandardItem(cat)
            f = header.font();
            f.setBold(True)
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
        h_llm = QHBoxLayout()
        self.cmb_llm = QComboBox()
        self.cmb_llm.addItems(settings.LLM_CATALOG.keys())
        h_llm.addWidget(self.cmb_llm)

        # self.lbl_llm_precision = QLabel("Mode:")
        # self.cmb_llm_precision = QComboBox()
        # h_llm.addWidget(self.lbl_llm_precision)
        # h_llm.addWidget(self.cmb_llm_precision)

        h_llm.addStretch(1)

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

        ##### USING GPU STATUS ICON #####
        status_widget = QWidget()
        status_layout = QHBoxLayout(status_widget)
        status_layout.setContentsMargins(0, 0, 0, 5)
        status_layout.setSpacing(6)

        from custom_widgets.PulsingDot import PulsingDot
        self.using_gpu_icon = PulsingDot(color="grey")
        label = QLabel("Using GPU")
        label.setStyleSheet("QLabel { margin-top: -3px;  }")

        status_layout.addWidget(self.using_gpu_icon)
        status_layout.addWidget(label)
        status_layout.addStretch()

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
        self.btn_chat.clicked.connect(
            #lambda: self._open_chat(self.cmb_llm.currentText(), self.cmb_llm_precision.currentData())
            lambda: self._open_chat(self.cmb_llm.currentText(), "4bit") 
        )
        self.btn_chat.setEnabled(False)
        v_ctrl.addWidget(self.btn_chat)

        grp_ctrl.setLayout(v_ctrl)
        main_right.addWidget(grp_ctrl)
        ##############################################################################################################################################
        # Aggregate Function UI
        ##############################################################################################################################################
        # -------------------- Aggregate Scorer --------------------
        grp_agg = QGroupBox("Aggregate Function")
        grid = QGridLayout()

        # row 0 –– control buttons
        self.btn_agg_start = QPushButton("Start")
        self.btn_agg_start.clicked.connect(self._agg_start)
        self.btn_agg_start.setEnabled(False)
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
        self.w_left = QWidget()
        self.w_left.setLayout(main_left)
        w_right = QWidget()
        w_right.setLayout(main_right)

        # Use 'Preferred' policy to allow the window to shrink 
        self.w_left.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        w_right.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        main_layout.addWidget(self.w_left, alignment=Qt.AlignmentFlag.AlignTop)
        main_layout.addWidget(w_right, alignment=Qt.AlignmentFlag.AlignTop)

        ################################################################
        # LLM add
        # self.cmb_llm.currentIndexChanged.connect(
        #     lambda: self._update_llm_options(self.cmb_llm, self.cmb_llm_precision, self.lbl_llm_precision)
        # )
        # self._update_llm_options(self.cmb_llm, self.cmb_llm_precision, self.lbl_llm_precision)

        ################################################################
        self._on_model_changed()

    def _build_ui_compact(self, main_layout):

        main_layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetMinimumSize)

        # ----------- Controls (start/stop/export/chat) ----------
        grp_ctrl = QGroupBox("Inference Controls")
        v_ctrl = QVBoxLayout(grp_ctrl)

        # GPU status
        status_widget = QWidget()
        status_layout = QHBoxLayout(status_widget)
        status_layout.setContentsMargins(0, 0, 0, 5)
        status_layout.setSpacing(6)

        from custom_widgets.PulsingDot import PulsingDot
        self.using_gpu_icon = PulsingDot(color="grey")
        label = QLabel("Using GPU")
        label.setStyleSheet("QLabel { margin-top: -3px;  }")

        status_layout.addWidget(self.using_gpu_icon)
        status_layout.addWidget(label)
        status_layout.addStretch()

        v_ctrl.addWidget(status_widget)

        # --- 2×2 Grid of main buttons ---
        grid_ctrl = QGridLayout()
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_export = QPushButton("Export")
        self.btn_chat = QPushButton("Chat with LLM")

        self.btn_start.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_export.clicked.connect(self._export)
        self.btn_chat.clicked.connect(self._open_llm_popup)  # NEW popup handler

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.btn_chat.setEnabled(False)

        grid_ctrl.addWidget(self.btn_start, 0, 0)
        grid_ctrl.addWidget(self.btn_stop, 0, 1)
        grid_ctrl.addWidget(self.btn_export, 1, 0)
        grid_ctrl.addWidget(self.btn_chat, 1, 1)

        v_ctrl.addLayout(grid_ctrl)
        main_layout.addWidget(grp_ctrl)

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
        for cat, items in settings.MODEL_CATELOG:
            header = QStandardItem(cat)
            f = header.font()
            f.setBold(True)
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
        main_layout.addWidget(grp_model)

        # -------------------- Additional configs --------------------------
        self.grp_cfg = CollapsibleGroupBox("Additional Configs")
        self.v_cfg = self.grp_cfg.content_layout()
        main_layout.addWidget(self.grp_cfg)

        # self.grp_cfg.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        # self.grp_cfg.content_layout().setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

        # -------------------- Screen Capture (collapsible) --------------------
        grp_cap = CollapsibleGroupBox("Screen Capture")
        v_cap = grp_cap.content_layout()

        # NOTE: Shows as popup instead of UI text
        # self.lbl_cap_hint = QLabel("Recommended Capture Size:")
        # self.lbl_cap_hint.setStyleSheet("color:red;")
        # v_cap.addWidget(self.lbl_cap_hint)

        btn_sel = QPushButton("Select Screen Region")
        btn_sel.clicked.connect(self._select_region)
        v_cap.addWidget(btn_sel)

        self.lbl_region = QLabel("Selected Region: NOT SET")
        self.lbl_region.setStyleSheet("color:grey;")
        v_cap.addWidget(self.lbl_region)

        main_layout.addWidget(grp_cap)

        ##############################################################################################################################################
        # Aggregate Function UI
        ##############################################################################################################################################

        # -------------------- Aggregate Function (collapsible) --------------------
        grp_agg = CollapsibleGroupBox("Aggregate Function")
        grid = QGridLayout()
        grp_agg.content_layout().addLayout(grid)

        self.btn_agg_start = QPushButton("Start")
        self.btn_agg_stop = QPushButton("Stop")
        self.btn_reset = QPushButton("Reset")
        self.btn_calib = QPushButton("Calibrate")
        self.btn_agg_export = QPushButton("Export")

        self.btn_agg_start.clicked.connect(self._agg_start)
        self.btn_agg_stop.clicked.connect(self._agg_stop)
        self.btn_reset.clicked.connect(self._on_reset)
        self.btn_calib.clicked.connect(self._calibrate_area)
        self.btn_agg_export.clicked.connect(self._agg_export)

        self.btn_agg_stop.setEnabled(False)
        self.btn_reset.setEnabled(False)
        self.btn_calib.setEnabled(False)
        self.btn_agg_export.setEnabled(False)
        self.btn_agg_start.setEnabled(False)

        grid.addWidget(self.btn_agg_start, 0, 0)
        grid.addWidget(self.btn_agg_stop, 0, 1)
        grid.addWidget(self.btn_reset, 0, 2)

        # second row (centered)
        h_row2 = QHBoxLayout()
        h_row2.addStretch()
        h_row2.addWidget(self.btn_calib)
        h_row2.addWidget(self.btn_agg_export)
        h_row2.addStretch()
        grid.addLayout(h_row2, 1, 0, 1, 3)  # span across 3 columns

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
        grid.addWidget(box, 2, 0, 1, 3)

        main_layout.addWidget(grp_agg)

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
        main_layout.addWidget(grp_out)

        main_layout.addStretch()

        self._on_model_changed()

    # ------------- UX ---------------
    def _select_region(self):

        if self.compact_mode:
            name = self.cmb_model.currentText()
            meta = settings.MODEL_METADATA[name]

            tile = meta["tile_size"]
            mpp = meta.get("mpp", 0.25)
            mag = math.ceil(0.25 / mpp * 40)

            dialog = QDialog(self.current_widget)
            dialog.setWindowTitle("Help")
            layout = QVBoxLayout(dialog)
            layout.addWidget(QLabel("<b>Recommended Capture Size:</b>"))
            layout.addWidget(QLabel(f"• {mag}× magnification"))
            layout.addWidget(QLabel(f"• Minimum {tile}×{tile}px"))
            layout.addWidget(
                QLabel("<i>(Smaller captures will be upscaled; larger ones will be sliced)</i><br>"))
            layout.addWidget(QLabel("<b>Please click OK and your next 2 clicks will be recorded.</b>"))
            layout.addWidget(QLabel("One for the top-left corner and one for the bottom-right corner."))

            button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            layout.addWidget(button_box)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)

            if dialog.exec() != QDialog.DialogCode.Accepted:
                return

        def get_mouse_pos():
            pos = []

            def _click(x, y, button, pressed):
                if pressed:
                    pos.append((x, y));
                    return False

            with mouse.Listener(on_click=_click) as l: l.join()
            return pos[0] if pos else (0, 0)

        x1, y1 = get_mouse_pos();
        x2, y2 = get_mouse_pos()
        self.selected_region = {
            "left": min(x1, x2), "top": min(y1, y2),
            "width": abs(x1 - x2), "height": abs(y1 - y2),
        }
        self.lbl_region.setText(
            f"Selected Region:\nWidth: {self.selected_region['width']:.0f} px\n"
            f"Height: {self.selected_region['height']:.0f} px")
        self.lbl_region.setStyleSheet("color:green;")
        self.btn_start.setEnabled(True)

    # ------------------------------------- Thread------------------------------------------------------
    def _start(self):
        # Display disclaimer if not suppressed by user settings
        if not self.settings.get("suppress_disclaimer", False):
            dialog = DisclaimerDialog(self.current_widget)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            if dialog.checkbox.isChecked():
                self.settings["suppress_disclaimer"] = True
                self._save_settings()

        if self.selected_region is None:
            QMessageBox.warning(self, "No region", "Please select a screen region first.")
            return
        if self.thread and self.thread.isRunning():
            self.thread.stop()
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

        self.btn_agg_start.setEnabled(True)
        self.btn_calib.setEnabled(True)

    def _stop(self):
        if self.thread:
            self.thread.stop()
            self.thread = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_chat.setEnabled(False)
        self.using_gpu_icon.set_color("grey")

        self._agg_stop()
        self.btn_agg_start.setEnabled(False)
        self.btn_calib.setEnabled(True)

    # ---------------------------------- Display----------------------------------
    def _update_display(self, frame, txt, metrics):
        self.latest_metrics = metrics
        self.latest_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB).copy()  # annotated
        self.latest_original = metrics.get("orig_img", self.latest_frame).copy()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR).copy()
        h, w, c = frame_bgr.shape

        # CENTRAL 1/3 × 1/3  CALIBRATION BOX
        if self.show_calib_box:
            bx = int(w / 3);
            by = int(h / 3)  # Box Size
            cx = (w - bx) // 2;
            cy = (h - by) // 2  # Put it in the middle
            p1, p2 = (cx, cy), (cx + bx, cy + by)
            bar = min(2, max(1, h // 140))

            # Transparent
            overlay = frame_bgr.copy()
            cv2.rectangle(overlay, p1, p2, (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.18, frame_bgr, 0.82, 0, frame_bgr)

            # 白描边 + 黑实线
            # cv2.rectangle(frame_bgr, (p1[0]-1, p1[1]-1), (p2[0]+1, p2[1]+1),
            #             (250, 250, 250), bar, lineType=cv2.LINE_AA)
            cv2.rectangle(frame_bgr, p1, p2, (250, 250, 250), bar, lineType=cv2.LINE_AA)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(frame_rgb.data, w, h, c * w, QImage.Format.Format_RGB888)
        if self.compact_mode:
            self.lbl_img.setPixmap(QPixmap.fromImage(qimg).scaled(225, 225, Qt.AspectRatioMode.KeepAspectRatio))
        else:
            self.lbl_img.setPixmap(QPixmap.fromImage(qimg).scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))
        self.lbl_res.setText(f"Result: {txt}")

        self.latest_frame = frame_rgb.copy()
        self.last_result = txt
        self.btn_export.setEnabled(True)

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
    def _open_chat(self, llm_name, llm_precision):
        if self.latest_frame is None:
            return

        self._stop()  # frees GPU memory so that we can load LLM
        self.btn_chat.setEnabled(True)

        dlg = LLMChatDialog(self.latest_frame, settings.LLM_CATALOG[llm_name], llm_precision, self)
        dlg.exec()

    # This is used for compact mode (instead of displaying on main GUI)
    # def _open_llm_popup(self):
    #     dialog = QDialog(self)
    #     dialog.setWindowTitle("Select GPT Options")

    #     layout = QVBoxLayout(dialog)
    #     cmb_llm = QComboBox()
    #     cmb_llm.addItems(settings.LLM_CATALOG.keys())
    #     layout.addWidget(cmb_llm)

    #     lbl_llm_precision = QLabel("Mode:")
    #     cmb_llm_precision = QComboBox()
    #     layout.addWidget(lbl_llm_precision)
    #     layout.addWidget(cmb_llm_precision)

    #     button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    #     button_box.accepted.connect(
    #         lambda: self._open_chat(cmb_llm.currentText(), cmb_llm_precision.currentData())
    #     )
    #     button_box.accepted.connect(dialog.accept)
    #     button_box.rejected.connect(dialog.reject)
    #     layout.addWidget(button_box)

    #     # LLM add
    #     cmb_llm.currentIndexChanged.connect(
    #         lambda: self._update_llm_options(cmb_llm, cmb_llm_precision, lbl_llm_precision)
    #     )
    #     self._update_llm_options(cmb_llm, cmb_llm_precision, lbl_llm_precision)

    #     dialog.exec()

    def _open_llm_popup(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Select GPT Model")

        layout = QVBoxLayout(dialog)
        cmb_llm = QComboBox()
        cmb_llm.addItems(settings.LLM_CATALOG.keys())
        layout.addWidget(QLabel("Select a model to chat with:"))
        layout.addWidget(cmb_llm)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        
        # This now directly calls _open_chat with "4bit"
        button_box.accepted.connect(
            lambda: self._open_chat(cmb_llm.currentText(), "4bit")
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.exec()


    # ------------------------ Helpers-----------------------------------------
    def _show_model_info(self):
        info = settings.MODEL_METADATA[self.cmb_model.currentText()].get("info", "No info.")
        QMessageBox.information(self, "Model Info", info)

    def _on_model_changed(self):
        """
        Stop the current classification thread before switching models.
        """
        if self.thread and self.thread.isRunning():
            self._stop()  # kill the old worker

        name = self.cmb_model.currentText()
        meta = settings.MODEL_METADATA[name]

        tile = meta["tile_size"]
        mpp = meta.get("mpp", 0.25)
        mag = math.ceil(0.25 / mpp * 40)
        if not self.compact_mode:
            self.lbl_cap_hint.setTextFormat(Qt.TextFormat.RichText) # For HTML rendering
            self.lbl_cap_hint.setText(
                f"""
                <b>Recommended Capture Size:</b><br>
                • {mag}× magnification<br>
                • Minimum {tile}×{tile}px<br>
                <i>(Smaller captures will be upscaled; larger ones will be sliced)</i><br><br>
                <b>Your next 2 clicks will be recorded:</b><br>
                Top-left corner & bottom-right corner
                """
            )
        else:
            # Text takes up too much space. Shows as popup in compact mode
            pass

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

        self.additional_config_inputs.clear()
        add_cfg = meta.get("additional_configs", {})
        self.grp_cfg.setVisible(bool(add_cfg))

        # Clear existing widgets
        clear_layout(self.v_cfg)

        for k, v in add_cfg.items():
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)  # optional
            lbl = QLabel(k)
            edt = QLineEdit(str(v))
            row_layout.addWidget(lbl)
            row_layout.addWidget(edt)
            self.v_cfg.addWidget(row_widget)  # <-- add widget, not layout
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
