# Main application for OnSight Pathology
#
#
# Run:
#   App main.py

import os
import sys
import logging
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

#os.environ["QT_SCALE_FACTOR"] = "1"
#os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
#os.environ["TQDM_DISABLE"] = "1"

# ---- Set DPI awareness BEFORE any GUI library imports ----
if sys.platform == "win32":
    try:
        import ctypes
        # Try Per-Monitor-V2 (Windows 10 1703+), fall back to Per-Monitor, then System DPI
        try:
            ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # PER_MONITOR_AWARE_V2
        except Exception:
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
            except Exception:
                ctypes.windll.user32.SetProcessDPIAware()
    except Exception as e:
        # Will be logged after crash_logging is imported
        _dpi_init_error = str(e)
    else:
        _dpi_init_error = None
else:
    _dpi_init_error = None
import crash_logging
import logging  
if _dpi_init_error:
    logging.warning(f"DPI awareness init failed: {_dpi_init_error}")
else:
    logging.info("DPI awareness set successfully")
import os
import sys

from datetime import datetime
from dataclasses import dataclass

import csv
import gc
import json
import math
import re
import time

import cv2
import numpy as np
import torch

from pynput import mouse

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRect, QSize
from PyQt6.QtGui import (
    QAction,
    QIcon,
    QImage,
    QKeySequence,
    QPixmap,
    QShortcut,
    QStandardItem,
    QStandardItemModel,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLayout,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

import settings
from utils import (
    build_precision_labels,
    get_gpu_memory,
    get_system_memory,
    load_model,
    resource_path,
)
from utils_clustering import clear_cluster_models_cache

from custom_widgets.AboutDialog import AboutDialog
from custom_widgets.cascade_widget import CascadeClusteringWidget  
from custom_widgets.CheckableComboBox import CheckableComboBox
from custom_widgets.CollapsibleGroupBox import CollapsibleGroupBox
from custom_widgets.DisclaimerDialog import DisclaimerDialog
from custom_widgets.LLMChatDialog import LLMChatDialog
from custom_widgets.ResizeImageDialog import ResizableImageDialog
from custom_widgets.mag_detector_widget import MagDetectorWidget
from custom_widgets.overlay_widget import OverlayWidget as ClusteringOverlay
from custom_widgets.overlay_widget_attention import OverlayWidget as HistomicsOverlay
from model_loader_thread import ModelLoaderThread

def run_llm_worker_if_requested() -> None:
    """If launched with --run-llm-worker, start worker process and exit."""
    if "--run-llm-worker" not in sys.argv:
        return

    sys.argv.remove("--run-llm-worker")
    import llm_worker_process  

    sys.exit(0)

run_llm_worker_if_requested()


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
    dynamic_sums: dict = None

    def __post_init__(self):
        if self.dynamic_sums is None:
            self.dynamic_sums = {}

    def reset(self):
        self.total_area_mm2 = self.conf_sum = self.mitosis = 0
        self.n_tiles = self.mib_pos = self.mib_total = 0
        
        self.cellularity_sum = 0.0
        self.dynamic_sums = {} 
    
    def __post_init__(self):
        if self.dynamic_sums is None:
            self.dynamic_sums = {}

    def reset(self):
        self.total_area_mm2 = self.conf_sum = self.mitosis = 0
        self.n_tiles = self.mib_pos = self.mib_total = 0
        
        self.cellularity_sum = 0.0
        self.dynamic_sums = {}

###############################################################################################################
# GUI Stylesh
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
    update_image = pyqtSignal(np.ndarray, str, dict)

    def __init__(self, ui_instance, model_name, preloaded=None):
        super().__init__()
        self.ui = ui_instance
        self.running = True
        self.metadata = settings.MODEL_METADATA[model_name]

        if preloaded is not None:
            # Use pre-loaded resources (skip slow HF download)
            self.model = preloaded["model"]
            self.process_region = preloaded["process_region_func"]
            self.using_gpu = preloaded["using_gpu"]
        else:
            # Fallback: original behavior (synchronous load)
            res = load_model(self.metadata)
            self.model = res["model"]
            self.process_region = res["process_region_func"]
            self.using_gpu = res["using_gpu"]

    @classmethod
    def from_loaded(cls, ui_instance, model_name, preloaded):
        return cls(ui_instance, model_name, preloaded=preloaded)

    # run() 和 stop() 保持不动
    def run(self):
        while self.running:
            try:
                t0 = time.time()
                extra_cfg = {}
                for k, w in self.ui.additional_config_inputs.items():
                    if isinstance(w, CheckableComboBox):
                        extra_cfg[k] = w.get_checked_items()
                    elif isinstance(w, QComboBox):
                        extra_cfg[k] = w.currentText()
                    elif isinstance(w, QCheckBox):
                        extra_cfg[k] = w.isChecked()
                    else:
                        extra_cfg[k] = w.text()
                extra_cfg['cascade_pipeline'] = getattr(self.ui, 'current_cascade_pipeline', [])

                # check mpp calibration status and update extra_cfg accordingly
                saved_mpp = self.ui.calibrated_mpp
                if saved_mpp and "mpp" in extra_cfg:
                    try:
                        if abs(float(extra_cfg["mpp"]) - saved_mpp) < 1e-5:
                            extra_cfg["is_calibrated"] = True
                        else:
                            extra_cfg["is_calibrated"] = False
                    except ValueError:
                        extra_cfg["is_calibrated"] = False
                else:
                    extra_cfg["is_calibrated"] = False

                frame, res_txt, metrics = self.process_region(
                    self.ui.selected_region,
                    model=self.model,
                    metadata=self.metadata,
                    additional_configs=extra_cfg,
                )
                res_txt += f"<br><span style='color:gray; font-size:8pt;'>({time.time() - t0:.4f}s)</span>"
                self.update_image.emit(frame, res_txt, metrics)
                #time.sleep(0.1)
            except torch.cuda.OutOfMemoryError as e:
                logging.error("CUDA out of memory in inference thread", exc_info=True)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                err_img = np.zeros((256, 256, 4), dtype=np.uint8)
                self.update_image.emit(
                    err_img,
                    "<b style='color:red'>GPU out of memory.</b><br>"
                    "Try a smaller capture region, or switch to CPU build.",
                    {}
                )
                time.sleep(2)

            except Exception as e:
                logging.critical("Inference thread exception", exc_info=True)
                err_img = np.zeros((256, 256, 4), dtype=np.uint8)
                err_short = f"{type(e).__name__}: {str(e)[:200]}"
                self.update_image.emit(
                    err_img,
                    f"<b style='color:red'>Error during inference:</b><br>"
                    f"<code>{err_short}</code><br>"
                    f"<small>Full details in log file.</small>",
                    {}
                )
                time.sleep(2)

    def stop(self):
        self.running = False
        self.wait()

        #free the model from GPU memory
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
        self.latest_frame = None  
        self.last_result = ""
        self.enlarged_window = None

        # Load persistent settings
        def get_user_settings_path(filename="settings.json"):
            local_appdata = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
            settings_dir = os.path.join(local_appdata, "OnSightPathology", "Settings")
            os.makedirs(settings_dir, exist_ok=True)
            return os.path.join(settings_dir, filename)

        self.settings_file = get_user_settings_path()
        self.settings = self._load_settings()


        # dictionary management for MPP (5x, 10x, 20x, 40x) >>>
        self.calibrated_mpps = self.settings.get("calibrated_mpps", {"5x": None, "10x": None, "20x": None, "40x": None})
        self.current_mag = self.settings.get("current_mag", "20x") 
        self.calibrated_mpp = self.calibrated_mpps.get(self.current_mag)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.central_layout = QVBoxLayout(self.central_widget)
        self.central_layout.setContentsMargins(0, 0, 0, 0) 
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

        self.cls_stats = {} 

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
            return {}  

    def _save_settings(self):
        """Saves current settings to the JSON file."""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def _update_global_mpp_label(self):
        """Updates the MPP label in the UI based on the currently selected magnification and its calibrated MPP value."""
        if not hasattr(self, 'lbl_global_mpp'): return
        mpp = self.calibrated_mpps.get(self.current_mag)
        if mpp:
            self.lbl_global_mpp.setText(f"MPP: {mpp:.3f} μm/px")
            self.lbl_global_mpp.setStyleSheet("color: #1abc9c; font-weight: bold;")
        else:
            self.lbl_global_mpp.setText("Not calibrated")
            self.lbl_global_mpp.setStyleSheet("color: grey;")

    def _on_global_mag_changed(self, text):
        """Triggered when the user switches 5x/10x/20x/40x on the left side"""
        self.current_mag = text
        self.settings["current_mag"] = text
        self.calibrated_mpp = self.calibrated_mpps.get(text)
        self._save_settings()
        self._update_global_mpp_label()
        

        if "mpp" in self.additional_config_inputs:
            if self.calibrated_mpp is not None:
                self.additional_config_inputs["mpp"].setText(f"{self.calibrated_mpp:.3f}")
            else:
                meta = settings.MODEL_METADATA.get(self.cmb_model.currentText(), {})
                self.additional_config_inputs["mpp"].setText(str(meta.get("mpp", 0.25)))


    # Mag detection and auto-sync related functions 
    def _on_auto_mag_clicked(self, checked):
        """Triggered when the user manually clicks the Auto-sync AI checkbox."""
        
        # Check if the user is trying to enable auto-sync without the detector running.
        if checked:
            if not hasattr(self, 'mag_widget') or not self.mag_widget.is_tracking:
                QMessageBox.warning(
                    self, 
                    "Detector Not Running", 
                    "Please start the 'Real-time Mag' detector first before enabling Auto-sync."
                )
                self.chk_auto_mag.blockSignals(True)
                self.chk_auto_mag.setChecked(False)
                self.chk_auto_mag.blockSignals(False)
                return 
                
        self.settings["auto_mag_sync"] = checked
        self._save_settings()
        
    
        if hasattr(self, 'cmb_global_mag'):
            self.cmb_global_mag.setEnabled(not checked)

    def _on_mag_tracking_changed(self, is_tracking):
        """Triggered when the mag detector starts or stops tracking. 
        Enables/disables the Auto-sync checkbox and shows/hides warnings as needed."""

        if hasattr(self, 'chk_auto_mag'):
            self.chk_auto_mag.setEnabled(is_tracking)
            
            if is_tracking:
                self.chk_auto_mag.setToolTip("Automatically sync AI detected magnification")
                was_auto = self.settings.get("auto_mag_sync", False)
                self.chk_auto_mag.setChecked(was_auto)
                if hasattr(self, 'cmb_global_mag'):
                    self.cmb_global_mag.setEnabled(not was_auto)
            else:
                self.chk_auto_mag.setToolTip("Please start the Real-time Mag detector first.")
                self.chk_auto_mag.setChecked(False)
                if hasattr(self, 'cmb_global_mag'):
                    self.cmb_global_mag.setEnabled(True)
                    
        if not is_tracking and hasattr(self, 'lbl_mag_warning'):
            self.lbl_mag_warning.hide()
      
    def _on_mag_detected(self, continuous_mag, pred_cls):
            match = re.search(r'\d+', str(pred_cls))
            if match:
                clean_cls = f"{match.group()}x"
            else:
                clean_cls = str(pred_cls).lower()

            if hasattr(self, 'supported_mags') and hasattr(self, 'lbl_mag_warning'):
                if hasattr(self, 'mag_widget') and self.mag_widget.is_tracking:
                    if clean_cls not in self.supported_mags:
                        req_str = " or ".join(self.supported_mags)
                        warning_html = (
                            f"<div style='line-height: 1.5;'>"
                            f"  <b style='font-size: 12pt; color: #ff4c4c;'>⚠️ Magnification Mismatch</b><br>"
                            f"  <span style='color: #ecf0f1; font-size: 10pt;'>Model requires: </span>"
                            f"  <b style='color: #1abc9c; font-size: 11pt;'>{req_str}</b><br>"
                            f"  <span style='color: #ecf0f1; font-size: 10pt;'>Current view: </span>"
                            f"  <b style='color: #f39c12; font-size: 11pt;'>{clean_cls}</b>"
                            f"</div>"
                        )
                        self.lbl_mag_warning.setText(warning_html)
                        self.lbl_mag_warning.show()
                    else:
                        self.lbl_mag_warning.hide()

            if not self.chk_auto_mag.isChecked() or not (hasattr(self, 'mag_widget') and self.mag_widget.is_tracking):
                return
                
            if hasattr(self, 'cmb_global_mag'):
                if self.cmb_global_mag.currentText() != clean_cls:
                    index = self.cmb_global_mag.findText(clean_cls)
                    if index >= 0:
                        self.cmb_global_mag.setCurrentIndex(index)

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
        self.btn_reset.setEnabled(False)  
        self.btn_agg_stop.setFocus()
        self.agg_records.clear()
        self.btn_agg_export.setEnabled(False)
        self.show_calib_box = False
        self.cls_stats.clear()

    def _update_overlay_params(self):
            """Pushes new spinbox values to the ACTIVE overlay widget in real-time."""
            if hasattr(self, 'current_active_overlay') and self.current_active_overlay:
                method = self.cmb_roi_method.currentText()
                
                if method == "Cluster":
                    self.current_active_overlay.patch_size = self.spin_patch.value()
                    self.current_active_overlay.n_clusters = self.spin_clusters.value()
                    self.current_active_overlay.sat_thresh = self.spin_sat.value()
                    self.current_active_overlay.val_thresh = self.spin_val.value()
                    self.current_active_overlay.tissue_thresh = self.spin_tissue.value()
                elif method == "Hotspot":
                    self.current_active_overlay.percentile = self.spin_percentile.value()
                    self.current_active_overlay.kernel_size = self.spin_kernel.value()

                      
    # Aggregate export function
    def _agg_export(self):
        if not self.agg_records:
            QMessageBox.information(
                self, "Nothing to export", "No tiles added yet.")
            return

        parent_dir = QFileDialog.getExistingDirectory(
            self, "Select export folder", "")
        if not parent_dir:
            return
        default_name = "Aggregate_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        custom_name, ok = QInputDialog.getText(
            self, "Export Folder Name",
            "Please enter a name for the export folder:",
            QLineEdit.EchoMode.Normal, default_name
        )
    
        if not ok or not custom_name.strip():
            return
            
        base_name = custom_name.strip()
        out_dir = os.path.join(parent_dir, base_name)
        counter = 1
        while os.path.exists(out_dir):
            out_dir = os.path.join(parent_dir, f"{base_name}_({counter})")
            counter += 1
        os.makedirs(out_dir, exist_ok=True)

        # save images and build rows for CSV
        rows = []
        all_clsset = set()
        for idx, rec in enumerate(self.agg_records, start=1):
            base = os.path.join(out_dir, f"{idx:03d}")  # 000,001…
            orig_path = base + ".png"
            anno_path = base + "_annotated.png"

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

        meta = settings.MODEL_METADATA[self.cmb_model.currentText()]
        agg_type = meta.get("aggregate_type", "classification")

        if agg_type == "cell_profiler" and rows:
            avg_row = {k: "" for k in rows[0].keys()}
            avg_row["file_orig"] = "OVERALL AVERAGE" 
            
            for col_name in rows[0].keys():
                if col_name not in ["file_orig", "file_annotated", "Class"]:
                    vals = [r[col_name] for r in rows if col_name in r and isinstance(r[col_name], (int, float))]
                    avg_row[col_name] = sum(vals) / len(vals) if vals else ""
                
            rows.append(avg_row)

        # wrte CSV
        csv_path = os.path.join(out_dir, "aggregate_metrics.csv")
        with open(csv_path, "w", newline="", encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        # Complete
        QMessageBox.information(
            self, "Export complete",
            f"Saved {len(rows)-1} images and metrics to:\n{out_dir}")

    # aggregate function triggered by space-bar shortcut and Add button
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
            current_mpp = self.calibrated_mpp if self.calibrated_mpp is not None else m.get("mpp", 0.25)
            mm2 = m["area_px"] * (current_mpp / 1000.0) ** 2

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

        # ---------- Single cell profiler ----------
        elif agg_type == "cell_profiler":
            reply = QMessageBox.question(
                self, "Add this tile?",
                (f"Cell Count: {m.get('cell_count', 0)}\n"
                 f"Cellularity: {m.get('cellularity_percent', 0):.0f} cells/mm²\n"
                 f"Tile area: {mm2:.3f} mm²\n\n"
                 "Add to aggregate?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes: return
            
            add_conf = add_mitos = add_pos = add_tot = add_cell = 0


            new_metrics = {"Tissue area (mm2)": mm2}
            for key, value in m.items():
                if isinstance(value, (int, float)):
                    new_metrics[key] = value

                    self.agg.dynamic_sums[key] = self.agg.dynamic_sums.get(key, 0.0) + value

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

        elif agg_type == "cell_profiler":
            n = self.agg.n_tiles
            # 从动态字典里安全读取你最想展示的几个核心指标
            avg_cell = self.agg.dynamic_sums.get('cellularity_percent', 0) / n if n else 0
            median_area = self.agg.dynamic_sums.get('Size.Area_median', 0) / n if n else 0
            median_circ = self.agg.dynamic_sums.get('Shape.Circularity_median', 0) / n if n else 0
            median_ecc = self.agg.dynamic_sums.get('Shape.Eccentricity_median', 0) / n if n else 0

            lines += [
                f"Avg Cellularity: {avg_cell:.2f} cells/mm²",
                f"Avg Nuclear Area: {median_area:.2f} μm²",
                f"Avg Circularity: {median_circ:.4f}",
                f"Avg Eccentricity: {median_ecc:.4f}",
                "",
                f"✓ Tracking {len(self.agg.dynamic_sums)} total features",
                f"  (Will be saved on Export)"
            ]
        return "\n".join(lines)
    def _calibrate_area(self):
        # ───────── First Click ─────────
        if not self.show_calib_box:
            if not self._calib_help_shown:
                QMessageBox.information(
                    self, "How to calibrate",
                    "Step 1: A translucent box will appear.\n"
                    "Step 2: In your slide viewer draw a box that exactly overlaps it\n"
                    "    and note the physical area.\n"
                    "Step 3: Click the 'Enter Area' button and input the measured value.")
                self._calib_help_shown = True

            self.show_calib_box = True
            QApplication.processEvents()
            self.btn_calib.setText("Enter area")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Calibrate Area")
        form = QVBoxLayout(dialog)
        form.addWidget(QLabel("Select magnification profile and enter area:"))
        
        h_layout = QHBoxLayout()
        
        # Magnification
        cmb_mag = QComboBox()
        cmb_mag.addItems(["5x", "10x", "20x", "40x"])
        if hasattr(self, 'cmb_global_mag'):
            cmb_mag.setCurrentText(self.cmb_global_mag.currentText()) 
            
        # SpinBox
        spin_val = QDoubleSpinBox()
        spin_val.setDecimals(4)
        spin_val.setMaximum(9999999.0)
        spin_val.setMinimum(0.0001)
        
        cmb_unit = QComboBox()
        cmb_unit.addItems(["μm²", "mm²"])
        
        h_layout.addWidget(cmb_mag)
        h_layout.addWidget(spin_val)
        h_layout.addWidget(cmb_unit)
        form.addLayout(h_layout)
        
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        form.addWidget(btn_box)

        spin_val.setFocus()
        spin_val.selectAll()

        if dialog.exec() == QDialog.DialogCode.Accepted and spin_val.value() > 0:
            val = spin_val.value()
            mag_key = cmb_mag.currentText()
            
            # convert to um² for consistent MPP calculation
            if cmb_unit.currentText() == "mm²":
                val_um2 = val * 1_000_000.0
            else:
                val_um2 = val

            if self.latest_frame is not None:
                h, w, c = self.latest_frame.shape
                bx = int(w / 3)
                by = int(h / 3)

                os_scale = self.devicePixelRatioF()
                physical_box_w = bx  / os_scale
                physical_box_h = by  / os_scale
                
                # calculate MPP: sqrt( physical area / pixel area )
                new_mpp = math.sqrt(val_um2 / (physical_box_w * physical_box_h))
                
                # save to the corresponding magnification profile 
                self.calibrated_mpps[mag_key] = new_mpp
                self.settings["calibrated_mpps"] = self.calibrated_mpps
                self._save_settings()
                
                if hasattr(self, 'cmb_global_mag'):
                    if self.cmb_global_mag.currentText() == mag_key:
                        self.calibrated_mpp = new_mpp
                        self._update_global_mpp_label()
                        if "mpp" in self.additional_config_inputs:
                            self.additional_config_inputs["mpp"].setText(f"{new_mpp:.3f}")
                    else:
                        self.cmb_global_mag.setCurrentText(mag_key)
                # ------------------------------
                
                QMessageBox.information(
                    self, "Calibration successful",
                    f"Saved to profile [{mag_key}]\n"
                    f"Calculated MPP = {new_mpp:.3f} μm/px"
                )
        else:
            QMessageBox.information(self, "Cancelled", "No valid area recorded.")

        # Close and Reset
        self.show_calib_box = False
        self.btn_calib.setText("Calibrate")    
   
    # LLM precision options update based on model metadata
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
    #Build UI
    ################################################################################################################################################
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

        btn_info = QPushButton("ℹ️")
        btn_info.setFixedSize(30, 30)
        btn_info.clicked.connect(self._show_model_info)
        h_model.addWidget(btn_info)

        grp_model.setLayout(h_model)
        main_left.addWidget(grp_model)
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

        self.mag_widget = MagDetectorWidget(get_region_callback=lambda: self.selected_region)
        self.mag_widget.mag_detected.connect(self._on_mag_detected)
        self.mag_widget.tracking_state_changed.connect(self._on_mag_tracking_changed)
        v_cap.addWidget(self.mag_widget)
        
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

        # --------------------Export button --------------------
        self.btn_export = QPushButton("Export")
        self.btn_export.clicked.connect(self._export)
        self.btn_export.setEnabled(False)
        v_ctrl.addWidget(self.btn_export)

        # --------------------Chat button --------------------
        self.btn_chat = QPushButton("Chat with LLM")
        self.btn_chat.clicked.connect(
            #lambda: self._open_chat(self.cmb_llm.currentText(), self.cmb_llm_precision.currentData())
            lambda: self._open_chat(self.cmb_llm.currentText(), "4bit") 
        )
        self.btn_chat.setEnabled(False)
        v_ctrl.addWidget(self.btn_chat)

        grp_ctrl.setLayout(v_ctrl)
        main_right.addWidget(grp_ctrl)

        grp_mag = QGroupBox("Active Magnification")
        h_mag = QHBoxLayout()
        
        self.cmb_global_mag = QComboBox()
        self.cmb_global_mag.addItems(["5x", "10x", "20x", "40x"])
        self.cmb_global_mag.setCurrentText(self.current_mag)
        self.cmb_global_mag.currentTextChanged.connect(self._on_global_mag_changed)
        
        self.lbl_global_mpp = QLabel()
        self._update_global_mpp_label()

        self.chk_auto_mag = QCheckBox("Auto-sync AI")
        self.chk_auto_mag.setStyleSheet("color: #3498db; font-weight: bold;")
        self.chk_auto_mag.setChecked(False) 
        self.settings["auto_mag_sync"] = False 
        
        self.chk_auto_mag.clicked.connect(self._on_auto_mag_clicked)


        h_mag.addWidget(self.cmb_global_mag)
        h_mag.addWidget(self.chk_auto_mag) 
        h_mag.addWidget(self.lbl_global_mpp)
        grp_mag.setLayout(h_mag)
        main_left.addWidget(grp_mag)

        # -------------------- Additional configs --------------------------
        self.grp_cfg = QGroupBox("Additional Configs")
        self.v_cfg = QVBoxLayout()
        self.grp_cfg.setLayout(self.v_cfg)
        main_left.addWidget(self.grp_cfg)


        grp_roi = QGroupBox("ROI Finder")
        v_roi = QVBoxLayout()

        h_roi_top = QHBoxLayout()
        h_roi_top.addWidget(QLabel("Method:"))
        self.cmb_roi_method = QComboBox()

        self.cmb_roi_method.addItems(["Hotspot", "Cluster"]) 
        self.cmb_roi_method.currentIndexChanged.connect(self._on_roi_method_changed)
        h_roi_top.addWidget(self.cmb_roi_method)
        h_roi_top.addSpacing(10)

        self.stacked_params = QStackedWidget()
        
        # --------------------Find ROI (Hotspot)--------------------
        page_hotspot = QWidget()
        page_hotspot.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        h_hotspot = QHBoxLayout(page_hotspot)
        h_hotspot.setContentsMargins(0, 0, 0, 0)
        
        self.lbl_percentile = QLabel("Percentile:")
        self.spin_percentile = QSpinBox()
        self.spin_percentile.setRange(10, 99)
        self.spin_percentile.setValue(60)
        self.spin_percentile.setSingleStep(5)
        self.spin_percentile.valueChanged.connect(self._update_overlay_params)

        
        # Kernel 
        self.lbl_kernel = QLabel("Kernel:")
        self.spin_kernel = QSpinBox()
        self.spin_kernel.setRange(1, 99) 
        self.spin_kernel.setValue(5)    
        self.spin_kernel.setSingleStep(2) 
        self.spin_kernel.valueChanged.connect(self._update_overlay_params)


        h_hotspot.addWidget(self.lbl_percentile)
        h_hotspot.addWidget(self.spin_percentile)
        h_hotspot.addWidget(self.lbl_kernel)
        h_hotspot.addWidget(self.spin_kernel)
        h_hotspot.addStretch() 
        self.stacked_params.addWidget(page_hotspot) 

        # --------------------Find ROI (Hotspot)--------------------
        page_cluster = QWidget()
        page_cluster.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored) 
        
        grid_cluster = QGridLayout(page_cluster)
        grid_cluster.setContentsMargins(0, 0, 0, 0)
        grid_cluster.setSpacing(8)
        
        # Row 0
        self.lbl_patch = QLabel("Patch:")
        self.spin_patch = QSpinBox()
        self.spin_patch.setRange(16, 256); self.spin_patch.setValue(36); self.spin_patch.setSingleStep(16)
        self.spin_patch.valueChanged.connect(self._update_overlay_params)
        
        self.lbl_clusters = QLabel("k:")
        self.spin_clusters = QSpinBox()
        self.spin_clusters.setRange(2, 20); self.spin_clusters.setValue(5)
        self.spin_clusters.valueChanged.connect(self._update_overlay_params)

        grid_cluster.addWidget(self.lbl_patch, 0, 0)
        grid_cluster.addWidget(self.spin_patch, 0, 1)
        grid_cluster.addWidget(self.lbl_clusters, 0, 2)
        grid_cluster.addWidget(self.spin_clusters, 0, 3)

        # Row 1
        self.spin_sat = QSpinBox()
        self.spin_sat.setRange(0, 255); self.spin_sat.setValue(10)
        self.spin_sat.valueChanged.connect(self._update_overlay_params)
        
        self.spin_val = QSpinBox()
        self.spin_val.setRange(0, 255); self.spin_val.setValue(250)
        self.spin_val.valueChanged.connect(self._update_overlay_params)
        
        self.spin_tissue = QDoubleSpinBox()
        self.spin_tissue.setRange(0.0, 1.0); self.spin_tissue.setSingleStep(0.05); self.spin_tissue.setValue(0.95)
        self.spin_tissue.valueChanged.connect(self._update_overlay_params)

        grid_cluster.addWidget(QLabel("Sat:"), 1, 0)
        grid_cluster.addWidget(self.spin_sat, 1, 1)
        grid_cluster.addWidget(QLabel("Val:"), 1, 2)
        grid_cluster.addWidget(self.spin_val, 1, 3)
        grid_cluster.addWidget(QLabel("Tissue:"), 1, 4)
        grid_cluster.addWidget(self.spin_tissue, 1, 5)
        
        grid_cluster.setColumnStretch(6, 1) 

        self.stacked_params.addWidget(page_cluster) 

        h_roi_top.addWidget(self.stacked_params)
        v_roi.addLayout(h_roi_top)

        h_overlay_ctrl = QHBoxLayout()
        self.btn_overlay_start = QPushButton("Find ROI")
        self.btn_overlay_pause = QPushButton("Pause")
        self.btn_overlay_stop = QPushButton("Stop")

        self.btn_overlay_start.setEnabled(False) 
        self.btn_overlay_pause.setEnabled(False)
        self.btn_overlay_stop.setEnabled(False)

        self.btn_overlay_start.clicked.connect(self._start_overlay)
        self.btn_overlay_pause.clicked.connect(self._pause_overlay)
        self.btn_overlay_stop.clicked.connect(self._stop_overlay)

        h_overlay_ctrl.addWidget(self.btn_overlay_start)
        h_overlay_ctrl.addWidget(self.btn_overlay_pause)
        h_overlay_ctrl.addWidget(self.btn_overlay_stop)
        v_roi.addLayout(h_overlay_ctrl)

        grp_roi.setLayout(v_roi)
        main_left.addWidget(grp_roi)

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

        
        # -------------------- Display output -----------------------------
        grp_out = QGroupBox("Inference Output")
        v_out = QVBoxLayout()

        self.lbl_img = QLabel()
        self.lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lbl_layout = QVBoxLayout(self.lbl_img)
        lbl_layout.setContentsMargins(0, 0, 0, 0)
        
        self.overlay_clustering = ClusteringOverlay(self.lbl_img)
        self.overlay_clustering.hide() 
        
        self.overlay_histomics = HistomicsOverlay(self.lbl_img)
        self.overlay_histomics.hide() 
        
        lbl_layout.addWidget(self.overlay_clustering, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        lbl_layout.addWidget(self.overlay_histomics, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        self.lbl_mag_warning = QLabel("⚠️ Warning: Magnification mismatch!")
        self.lbl_mag_warning.setStyleSheet("""
            background-color: rgba(30, 40, 50, 220); 
            padding: 8px 14px; 
            border: 1px solid #ff4c4c; 
            border-radius: 6px;
        """)
        self.lbl_mag_warning.hide() 
        lbl_layout.addWidget(self.lbl_mag_warning, alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft)

        self.overlay_clustering.worker.finished.connect(self._on_overlay_worker_finished)
        self.overlay_histomics.worker.finished.connect(self._on_overlay_worker_finished)
        
        self.current_active_overlay = None 

        v_out.addWidget(self.lbl_img)
        
        self.lbl_res = QLabel("Result:")
        v_out.addWidget(self.lbl_res)

        self.btn_enlarge = QPushButton("Expanded View 🔍")
        self.btn_enlarge.setEnabled(False)
        self.btn_enlarge.clicked.connect(self._open_enlarged_view)
        v_out.addWidget(self.btn_enlarge)

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

        self._on_roi_method_changed(0)
        self._on_model_changed()
    ################################################################################################################################################
    #Build UI Compact
    ################################################################################################################################################
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

        grp_mag = CollapsibleGroupBox("Active Magnification")
        h_mag = QHBoxLayout()
        
        self.cmb_global_mag = QComboBox()
        self.cmb_global_mag.addItems(["5x", "10x", "20x", "40x"])
        self.cmb_global_mag.setCurrentText(self.current_mag)
        self.cmb_global_mag.currentTextChanged.connect(self._on_global_mag_changed)
        
        self.lbl_global_mpp = QLabel()
        self._update_global_mpp_label()

        self.chk_auto_mag = QCheckBox("Auto-sync AI")
        self.chk_auto_mag.setStyleSheet("color: #3498db; font-weight: bold;")
        self.chk_auto_mag.setChecked(self.settings.get("auto_mag_sync", False)) 
        self.chk_auto_mag.clicked.connect(self._on_auto_mag_clicked)

        h_mag.addWidget(self.cmb_global_mag)
        h_mag.addWidget(self.chk_auto_mag) 
        h_mag.addWidget(self.lbl_global_mpp)
        grp_mag.content_layout().addLayout(h_mag)
        main_layout.addWidget(grp_mag)

        # -------------------- Additional configs --------------------------
        self.grp_cfg = CollapsibleGroupBox("Additional Configs")
        self.v_cfg = self.grp_cfg.content_layout()
        main_layout.addWidget(self.grp_cfg)

        # -------------------- Screen Capture (collapsible) --------------------
        grp_cap = CollapsibleGroupBox("Screen Capture")
        v_cap = grp_cap.content_layout()

        btn_sel = QPushButton("Select Screen Region")
        btn_sel.clicked.connect(self._select_region)
        v_cap.addWidget(btn_sel)

        self.mag_widget = MagDetectorWidget(get_region_callback=lambda: self.selected_region)
        self.mag_widget.mag_detected.connect(self._on_mag_detected)
        self.mag_widget.tracking_state_changed.connect(self._on_mag_tracking_changed)
        v_cap.addWidget(self.mag_widget)
        # +++++++++++++++++++++++++++++++++++++++++
        self.lbl_region = QLabel("Selected Region: NOT SET")
        self.lbl_region.setStyleSheet("color:grey;")
        v_cap.addWidget(self.lbl_region)

        main_layout.addWidget(grp_cap)

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

        h_row2 = QHBoxLayout()
        h_row2.addStretch()
        h_row2.addWidget(self.btn_calib)
        h_row2.addWidget(self.btn_agg_export)
        h_row2.addStretch()
        grid.addLayout(h_row2, 1, 0, 1, 3)

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

        # -------------------- Display output-----------------------------
        grp_out = QGroupBox("Inference Output")
        v_out = QVBoxLayout()

        self.lbl_img = QLabel()
        self.lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lbl_layout = QVBoxLayout(self.lbl_img)
        lbl_layout.setContentsMargins(0, 0, 0, 0)
        
        self.overlay_clustering = ClusteringOverlay(self.lbl_img)
        self.overlay_clustering.hide() 
        
        self.overlay_histomics = HistomicsOverlay(self.lbl_img)
        self.overlay_histomics.hide() 
        
        lbl_layout.addWidget(self.overlay_clustering, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        lbl_layout.addWidget(self.overlay_histomics, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        
        self.lbl_mag_warning = QLabel("⚠️ Warning: Magnification mismatch!")
        self.lbl_mag_warning.setStyleSheet("""
            background-color: rgba(30, 40, 50, 220); 
            padding: 8px 14px; 
            border: 1px solid #ff4c4c; 
            border-radius: 6px;
        """)
        self.lbl_mag_warning.hide() 
        lbl_layout.addWidget(self.lbl_mag_warning, alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft)


        self.overlay_clustering.worker.finished.connect(self._on_overlay_worker_finished)
        self.overlay_histomics.worker.finished.connect(self._on_overlay_worker_finished)


        self.current_active_overlay = None 

        v_out.addWidget(self.lbl_img)
        
        # -------------------- ROI Finder ------------------------------
        grp_roi = QGroupBox("ROI Finder")
        v_roi = QVBoxLayout()

        h_roi_top = QHBoxLayout()
        h_roi_top.addWidget(QLabel("Method:"))
        
        self.cmb_roi_method = QComboBox()
        self.cmb_roi_method.addItems(["Hotspot", "Cluster"]) 
        self.cmb_roi_method.currentIndexChanged.connect(self._on_roi_method_changed)
        h_roi_top.addWidget(self.cmb_roi_method)
        h_roi_top.addSpacing(10)

        self.stacked_params = QStackedWidget()
        
        # --- 1：Hotspot---
        page_hotspot = QWidget()
        page_hotspot.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        h_hotspot = QHBoxLayout(page_hotspot)
        h_hotspot.setContentsMargins(0, 0, 0, 0)
        
        self.lbl_percentile = QLabel("Percentile:")
        self.spin_percentile = QSpinBox()
        self.spin_percentile.setRange(10, 99)
        self.spin_percentile.setValue(60)
        self.spin_percentile.setSingleStep(5)
        self.spin_percentile.valueChanged.connect(self._update_overlay_params)

        self.lbl_kernel = QLabel("Kernel:")
        self.spin_kernel = QSpinBox()
        self.spin_kernel.setRange(1, 99) 
        self.spin_kernel.setValue(5)      
        self.spin_kernel.setSingleStep(2) 
        self.spin_kernel.valueChanged.connect(self._update_overlay_params)
        
        h_hotspot.addWidget(self.lbl_percentile)
        h_hotspot.addWidget(self.spin_percentile)
        h_hotspot.addWidget(self.lbl_kernel)
        h_hotspot.addWidget(self.spin_kernel)
        h_hotspot.addStretch() 
        self.stacked_params.addWidget(page_hotspot) 

        # --- 2：Cluster ---
        page_cluster = QWidget()
        page_cluster.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored) 
        
        grid_cluster = QGridLayout(page_cluster)
        grid_cluster.setContentsMargins(0, 0, 0, 0)
        grid_cluster.setSpacing(8) 
        
        self.lbl_patch = QLabel("Patch:")
        self.spin_patch = QSpinBox()
        self.spin_patch.setRange(16, 256); self.spin_patch.setValue(36); self.spin_patch.setSingleStep(16)
        self.spin_patch.valueChanged.connect(self._update_overlay_params)
        
        self.lbl_clusters = QLabel("k:")
        self.spin_clusters = QSpinBox()
        self.spin_clusters.setRange(2, 20); self.spin_clusters.setValue(5)
        self.spin_clusters.valueChanged.connect(self._update_overlay_params)

        grid_cluster.addWidget(self.lbl_patch, 0, 0)
        grid_cluster.addWidget(self.spin_patch, 0, 1)
        grid_cluster.addWidget(self.lbl_clusters, 0, 2)
        grid_cluster.addWidget(self.spin_clusters, 0, 3)

        self.spin_sat = QSpinBox()
        self.spin_sat.setRange(0, 255); self.spin_sat.setValue(10)
        self.spin_sat.valueChanged.connect(self._update_overlay_params)
        
        self.spin_val = QSpinBox()
        self.spin_val.setRange(0, 255); self.spin_val.setValue(250)
        self.spin_val.valueChanged.connect(self._update_overlay_params)
        
        self.spin_tissue = QDoubleSpinBox()
        self.spin_tissue.setRange(0.0, 1.0); self.spin_tissue.setSingleStep(0.05); self.spin_tissue.setValue(0.95)
        self.spin_tissue.valueChanged.connect(self._update_overlay_params)

        grid_cluster.addWidget(QLabel("Sat:"), 1, 0)
        grid_cluster.addWidget(self.spin_sat, 1, 1)
        grid_cluster.addWidget(QLabel("Val:"), 1, 2)
        grid_cluster.addWidget(self.spin_val, 1, 3)
        grid_cluster.addWidget(QLabel("Tissue:"), 1, 4)
        grid_cluster.addWidget(self.spin_tissue, 1, 5)
        grid_cluster.setColumnStretch(6, 1) 

        self.stacked_params.addWidget(page_cluster) 

        h_roi_top.addWidget(self.stacked_params)
        v_roi.addLayout(h_roi_top)

        h_overlay_ctrl = QHBoxLayout()
        self.btn_overlay_start = QPushButton("Find ROI")
        self.btn_overlay_pause = QPushButton("Pause")
        self.btn_overlay_stop = QPushButton("Stop")

        self.btn_overlay_start.setEnabled(False) 
        self.btn_overlay_pause.setEnabled(False)
        self.btn_overlay_stop.setEnabled(False)

        self.btn_overlay_start.clicked.connect(self._start_overlay)
        self.btn_overlay_pause.clicked.connect(self._pause_overlay)
        self.btn_overlay_stop.clicked.connect(self._stop_overlay)

        h_overlay_ctrl.addWidget(self.btn_overlay_start)
        h_overlay_ctrl.addWidget(self.btn_overlay_pause)
        h_overlay_ctrl.addWidget(self.btn_overlay_stop)
        v_roi.addLayout(h_overlay_ctrl)

        grp_roi.setLayout(v_roi)
        v_out.addWidget(grp_roi)

        self.lbl_res = QLabel("Result:")
        v_out.addWidget(self.lbl_res)

        self.btn_enlarge = QPushButton("Expanded View 🔍")
        self.btn_enlarge.setEnabled(False)
        self.btn_enlarge.clicked.connect(self._open_enlarged_view)
        v_out.addWidget(self.btn_enlarge)

        grp_out.setLayout(v_out)
        main_layout.addWidget(grp_out)

        main_layout.addStretch()
        
        self._on_roi_method_changed(0)
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

    # -------------------- Overlay Controls ---------------------------------
    def _start_overlay(self):
        if hasattr(self, 'current_active_overlay') and self.current_active_overlay:
        
            method = self.cmb_roi_method.currentText()
            if method == "Cluster":
                self.current_active_overlay.patch_size = self.spin_patch.value()
                self.current_active_overlay.n_clusters = self.spin_clusters.value()
            elif method == "Hotspot":
                self.current_active_overlay.percentile = self.spin_percentile.value()

            self.current_active_overlay.start()
            if (not self.thread or not self.thread.isRunning()) and self.latest_frame is not None:
                self.current_active_overlay.process_frame(self.latest_frame)
            self.btn_overlay_start.setEnabled(False)
            self.btn_overlay_pause.setEnabled(True)
            self.btn_overlay_stop.setEnabled(True)
            self.btn_overlay_pause.setText("Pause")

    def _pause_overlay(self):
        if hasattr(self, 'current_active_overlay') and self.current_active_overlay:
            if self.current_active_overlay.state == "RUNNING":
                self.current_active_overlay.pause()
                self.btn_overlay_pause.setText("Resume")
            else:
                self.current_active_overlay.state = "RUNNING"
                self.btn_overlay_pause.setText("Pause")

    def _stop_overlay(self):
            if hasattr(self, 'current_active_overlay') and self.current_active_overlay:
                self.current_active_overlay.stop()
                self.btn_overlay_start.setEnabled(True)
                self.btn_overlay_pause.setEnabled(False)
                self.btn_overlay_stop.setEnabled(False)
                self.btn_overlay_pause.setText("Pause")
                
                self.latest_overlay_result = None 
                if getattr(self, 'enlarged_window', None) and self.enlarged_window.isVisible():
                    self.enlarged_window.hide_overlay()

    # ------------------------Thread---------------------------
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
        self._start_with_progress(model_name)

        if hasattr(self, 'btn_overlay_start'):
            self.btn_overlay_start.setEnabled(True)

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_chat.setEnabled(True)

        self.btn_agg_start.setEnabled(True)
        self.btn_calib.setEnabled(True)

    def _stop(self):
        if self.thread:
            self.thread.stop()
            self.thread = None

        if hasattr(self, 'btn_overlay_start'):
            self.btn_overlay_start.setEnabled(False) 
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_chat.setEnabled(False)
        self.using_gpu_icon.set_color("grey")

        self._agg_stop()
        self.btn_agg_start.setEnabled(False)
        self.btn_calib.setEnabled(True)

    # ---------------------------- Display----------------------------------
    def _update_display(self, frame, txt, metrics):
        self.latest_metrics = metrics
        self.latest_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB).copy()  # annotated
        self.latest_original = metrics.get("orig_img", self.latest_frame).copy()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR).copy()
        h, w, c = frame_bgr.shape
        attn_toggle = self.additional_config_inputs.get("Show ViT Attention Map")
        
        if attn_toggle and hasattr(attn_toggle, 'isChecked') and attn_toggle.isChecked():
            if "attention_map" in metrics:
                attn_map = metrics["attention_map"]
                heatmap = cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_JET)
                frame_bgr = cv2.addWeighted(heatmap, 0.4, frame_bgr, 0.6, 0)

        # CENTRAL 1/3 × 1/3  CALIBRATION BOX
        if self.show_calib_box:
            bx = int(w / 3);4
            by = int(h / 3)  
            cx = (w - bx) // 2;
            cy = (h - by) // 2 
            p1, p2 = (cx, cy), (cx + bx, cy + by)
            bar = min(2, max(1, h // 140))
            overlay = frame_bgr.copy()
            cv2.rectangle(overlay, p1, p2, (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.18, frame_bgr, 0.82, 0, frame_bgr)

            cv2.rectangle(frame_bgr, p1, p2, (250, 250, 250), bar, lineType=cv2.LINE_AA)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(frame_rgb.data, w, h, c * w, QImage.Format.Format_RGB888)

        if self.compact_mode:
            self.lbl_img.setPixmap(QPixmap.fromImage(qimg).scaled(225, 225, Qt.AspectRatioMode.KeepAspectRatio))
        else:
            self.lbl_img.setPixmap(QPixmap.fromImage(qimg).scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))
        self.lbl_res.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_res.setText(txt) 



        
        self.latest_frame = frame_rgb.copy()
        if hasattr(self, 'current_active_overlay') and self.current_active_overlay:
            self.current_active_overlay.process_frame(self.latest_frame)

        self.last_result = txt
        self.btn_export.setEnabled(True)

        if hasattr(self, 'btn_enlarge'):
            self.btn_enlarge.setEnabled(True)
        if getattr(self, 'enlarged_window', None) and self.enlarged_window.isVisible():
            self.enlarged_window.set_image(self.latest_frame)
            if hasattr(self.enlarged_window, 'set_text'):
                self.enlarged_window.set_text(txt)

    # ---------------------------- Export------------------------------------------
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

    def _open_enlarged_view(self):
        if self.latest_frame is None:
            return
        
        if self.enlarged_window is None or not self.enlarged_window.isVisible():
            self.enlarged_window = ResizableImageDialog(None) 
            self.enlarged_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose) 
            self.enlarged_window.destroyed.connect(self._on_enlarged_window_closed)
            self.enlarged_window.show() 
            
        self.enlarged_window.set_image(self.latest_frame)
        if hasattr(self.enlarged_window, 'set_text'):
            self.enlarged_window.set_text(self.last_result)
            
        if hasattr(self, 'current_active_overlay') and self.current_active_overlay and self.current_active_overlay.isVisible():
            if getattr(self, 'latest_overlay_result', None) is not None:
                self.enlarged_window.set_overlay(self.latest_overlay_result)
        
        self.enlarged_window.activateWindow()

    def _on_enlarged_window_closed(self):
        self.enlarged_window = None

    # ---------------------------- Chat-------------------------------------------------
    def _open_chat(self, llm_name, llm_precision):
        if self.latest_frame is None:
            return

        self.btn_chat.setEnabled(True)

        dlg = LLMChatDialog(self.latest_frame, settings.LLM_CATALOG[llm_name], llm_precision, self)
        dlg.exec()



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

    def _on_roi_method_changed(self, index):
            method = self.cmb_roi_method.currentText()
            
            self._stop_overlay()
            
            if method == "Hotspot":
                self.current_active_overlay = self.overlay_histomics
                if hasattr(self, 'stacked_params'):
                    self.stacked_params.setCurrentIndex(0)
            else: # "Cluster"
                self.current_active_overlay = self.overlay_clustering
                if hasattr(self, 'stacked_params'):
                    self.stacked_params.setCurrentIndex(1)
                    
            if hasattr(self, 'stacked_params'):
                for i in range(self.stacked_params.count()):
                    widget = self.stacked_params.widget(i)
                    if i == self.stacked_params.currentIndex():
                        widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
                    else:
                        widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
                self.stacked_params.adjustSize()
                

            if hasattr(self, '_update_overlay_params'):
                self._update_overlay_params()

    def _on_overlay_worker_finished(self, result_rgb):

            self.latest_overlay_result = result_rgb.copy() 
            
            if getattr(self, 'enlarged_window', None) and self.enlarged_window.isVisible():
                self.enlarged_window.set_overlay(result_rgb)

        # loading model with progress dialog
    def _start_with_progress(self, model_name: str):
        from custom_widgets.LoadingDialog import LoadingDialog

        self._progress_dlg = LoadingDialog(
            title="Loading Model",
            model_name=model_name,
            parent=self
        )
        self._loader = ModelLoaderThread(model_name, parent=self)
        self._was_cancelled = False

        def on_progress(text, pct, cur_b, tot_b):
            if not self._progress_dlg:
                return
            self._progress_dlg.set_status(text)
            self._progress_dlg.set_progress(pct, cur_b, tot_b)

        def cleanup():
            """Close dialog and reset state regardless of outcome."""
            if self._progress_dlg:
                self._progress_dlg.accept()
                self._progress_dlg = None
            self._loader = None
            self.btn_start.setEnabled(True)

        def on_ok(res):
            if self._was_cancelled:
                cleanup()
                return
            cleanup()
            self.thread = ClassificationThread.from_loaded(self, model_name, res)
            self.thread.update_image.connect(self._update_display)
            self.thread.start()
            self.using_gpu_icon.set_color("green" if self.thread.using_gpu else "red")

            if hasattr(self, 'btn_overlay_start'):
                self.btn_overlay_start.setEnabled(True)
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.btn_chat.setEnabled(True)
            self.btn_agg_start.setEnabled(True)
            self.btn_calib.setEnabled(True)

        def on_fail(short, tb):
            cleanup()
            if self._was_cancelled:
                return

            hint = ""
            sl = short.lower()
            # if "huggingface" in sl or "getaddrinfo" in sl or "localentrynotfound" in sl or "connectionerror" in sl:
            #     hint = ("<b>Model weights could not be downloaded.</b><br>"
            #             "• Check your internet connection<br>"
            #             "• huggingface.co may be blocked on your network<br>"
            #             "• Some institutional firewalls require a proxy")
            # elif "out of memory" in sl or "cuda" in sl:
            #     hint = "<b>GPU initialization failed.</b><br>Try the CPU build or a smaller model."
            # elif "modulenotfound" in sl or "no module named" in sl:
            #     hint = ("<b>A required dependency is missing.</b><br>"
            #             "This is a packaging bug. Please report it so we can fix it.")

            from crash_logging import show_error_dialog
            show_error_dialog(
                title=f"Could not load '{model_name}'",
                short_msg=short,
                details=tb,
                hint_html=hint,
                parent=self
            )

        def on_finished():
            """QThread.finished — emitted when run() returns, even on cancel/abort."""
            if self._was_cancelled and self._progress_dlg:
                cleanup()

        def on_cancel():
            self._was_cancelled = True
            if self._loader and self._loader.isRunning():
                self._loader.request_abort()
            # The dialog stays open showing "Cancelling…" until on_finished fires.

        self._loader.progress.connect(on_progress)
        self._loader.finished_ok.connect(on_ok)
        self._loader.failed.connect(on_fail)
        self._loader.finished.connect(on_finished)   # ← Qt 自带 signal
        self._progress_dlg.cancelled.connect(on_cancel)

        self.btn_start.setEnabled(False)
        self._loader.start()
        self._progress_dlg.show()
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
        try:
            clear_cluster_models_cache(keep=None)
        except ImportError:
            pass
        name = self.cmb_model.currentText()
        meta = settings.MODEL_METADATA[name]

        tile = meta["tile_size"]
        mpp = meta.get("mpp", 0.25)
        mag = math.ceil(0.25 / mpp * 40)

        self.supported_mags = meta.get("supported_mags", [f"{mag}x"])
        mag_display = " or ".join(self.supported_mags) 
        
        if hasattr(self, 'lbl_mag_warning'):
            self.lbl_mag_warning.hide() 
        if not self.compact_mode:
            self.lbl_cap_hint.setTextFormat(Qt.TextFormat.RichText) 
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
                    clear_layout(child_layout)
        self.additional_config_inputs.clear()
        self.additional_config_rows = {} 
        add_cfg = meta.get("additional_configs", {})
        self.grp_cfg.setVisible(bool(add_cfg))

        clear_layout(self.v_cfg)

        for k, v in add_cfg.items():
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)  # optional
            lbl = QLabel(k)

            if "(Multi-Select)" in k and isinstance(v, list):
                edt = CheckableComboBox()
                edt.addItems([str(item) for item in v], checked_by_default=False)
            elif isinstance(v, list):
                edt = QComboBox()
                edt.addItems([str(item) for item in v])
            elif isinstance(v, bool):
                edt = QCheckBox()
                edt.setChecked(v)
            else:
                if k.lower() == "mpp" and self.calibrated_mpp is not None:
                    edt = QLineEdit(f"{self.calibrated_mpp:.5f}")
                else:
                    edt = QLineEdit(str(v))


            row_layout.addWidget(lbl)
            row_layout.addWidget(edt)
            self.v_cfg.addWidget(row_widget)  
            self.additional_config_inputs[k] = edt
            self.additional_config_rows[k] = row_widget

        if "Clustering Method" in self.additional_config_inputs:
            cmb_clustering = self.additional_config_inputs["Clustering Method"]
            
            row_context_scale = self.additional_config_rows.get("Context Scale")
            row_k = self.additional_config_rows.get("Clusters (k)")
            row_img_type = self.additional_config_rows.get("Image Type")
            row_feat_he = self.additional_config_rows.get("H&E Features (Multi-Select)")
            row_feat_muscle = self.additional_config_rows.get("Muscle Features (Multi-Select)")

            he_feats = meta.get("additional_configs", {}).get("H&E Features (Multi-Select)", [])
            mu_feats = meta.get("additional_configs", {}).get("Muscle Features (Multi-Select)", [])
            
            self.cascade_widget = CascadeClusteringWidget(he_feats)
            self.v_cfg.addSpacing(5)
            self.v_cfg.addWidget(self.cascade_widget)
            self.cascade_widget.hide() 
            
            self.current_cascade_pipeline = []
            self.cascade_widget.pipeline_updated.connect(lambda p: setattr(self, 'current_cascade_pipeline', p))

            def update_clustering_ui():
                method_text = cmb_clustering.currentText()
                img_type_text = ""
                if "Image Type" in self.additional_config_inputs:
                    img_type_text = self.additional_config_inputs["Image Type"].currentText()

                if row_context_scale:
                    row_context_scale.setVisible(method_text in ["Single Cell Clustering", "Region Clustering"])
                if row_k:
                    row_k.setVisible(method_text not in ["None", "Hierarchical Gating"])
                
                is_custom = (method_text == "Customized Features")
                if row_img_type: row_img_type.setVisible(is_custom)
                if row_feat_he: row_feat_he.setVisible(is_custom and img_type_text == "H&E Cell Analysis")
                if row_feat_muscle: row_feat_muscle.setVisible(is_custom and img_type_text == "Muscle Fiber Typing")

                if method_text == "Hierarchical Gating":
                    self.cascade_widget.show()
                    self.cascade_widget.emit_pipeline() 
                else:
                    self.cascade_widget.hide()

            cmb_clustering.currentTextChanged.connect(lambda _: update_clustering_ui())
            if "Image Type" in self.additional_config_inputs:
                self.additional_config_inputs["Image Type"].currentTextChanged.connect(lambda _: update_clustering_ui())
            
            update_clustering_ui()

        if "Refine Cell Subpopulations" in self.additional_config_inputs:
            cmb_refine = self.additional_config_inputs["Refine Cell Subpopulations"]
            
            row_refine_preview = self.additional_config_rows.get("Show Positives Only")
            row_refine_feats_multi = self.additional_config_rows.get("Refine Features (Multi-Select)")
            row_refine_feat_single = self.additional_config_rows.get("Refine Feature")
            row_refine_k = self.additional_config_rows.get("K Clusters")
            row_refine_target_multi = self.additional_config_rows.get("Select target groups (Multi-Select)")
            row_refine_cutoff = self.additional_config_rows.get("Threshold Cutoff")

            k_input = self.additional_config_inputs.get("K Clusters")
            target_cmb = self.additional_config_inputs.get("Select target groups (Multi-Select)")

            if isinstance(target_cmb, CheckableComboBox):
                target_cmb.placeholder_base = "groups"

            if k_input and target_cmb and isinstance(target_cmb, CheckableComboBox):
                def update_target_groups():
                    try:
                        text = k_input.text() if hasattr(k_input, 'text') else str(k_input.value())
                        k_val = int(text)
                    except:
                        k_val = 2
                    
                    k_val = max(2, min(k_val, 15))
                    checked_items = target_cmb.get_checked_items()
                    
                    target_cmb.blockSignals(True)
                    try:
                        target_cmb.clear_items()
                        
                        new_items = [f"Group{i}" for i in range(k_val)]
                        target_cmb.addItems(new_items, checked_by_default=False)
                        
                        for i in range(target_cmb.model().rowCount()):
                            item = target_cmb.model().item(i)
                            if item and item.text() in checked_items:
                                item.setCheckState(Qt.CheckState.Checked)
                    finally:
                        target_cmb.blockSignals(False)
                        if hasattr(target_cmb, '_update_display_text'):
                            target_cmb._update_display_text()

                if hasattr(k_input, 'textChanged'):
                    k_input.textChanged.connect(lambda _: update_target_groups())
                
                update_target_groups()

            def update_refine_ui():
                method = cmb_refine.currentText()
                is_km = (method == "K-Means")
                is_th = (method == "Threshold")
                if row_refine_preview: row_refine_preview.setVisible(is_km)
                if row_refine_feats_multi: row_refine_feats_multi.setVisible(is_km)
                if row_refine_k: row_refine_k.setVisible(is_km)
                if row_refine_target_multi: row_refine_target_multi.setVisible(is_km)
                if row_refine_feat_single: row_refine_feat_single.setVisible(is_th)
                if row_refine_cutoff: row_refine_cutoff.setVisible(is_th)

            cmb_refine.currentTextChanged.connect(lambda _: update_refine_ui())
            update_refine_ui()
      
# -----------------------------Close----------------------------------------
    def closeEvent(self, e):
        self._stop()
        if hasattr(self, 'mag_widget'):
            self.mag_widget.close_threads()
        if hasattr(self, 'mag_widget_compact'):
            self.mag_widget_compact.close_threads()
        if getattr(self, 'enlarged_window', None):
            self.enlarged_window.close()
        e.accept()

def main():
    app = QApplication(sys.argv)
    win = ImageClassificationApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()