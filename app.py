import logging
import os
import sys

from custom_widgets.mag_detector_widget import MagDetectorWidget
from custom_widgets.overlay_widget import OverlayWidget as ClusteringOverlay
from custom_widgets.overlay_widget_attention import OverlayWidget as HistomicsOverlay
from utils_clustering import clear_cluster_models_cache 
from custom_widgets.CheckableComboBox import CheckableComboBox
# NOTE: ONLY FOR CPU EXE
if 0:
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
    QApplication, QSpinBox, QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox,
    QLineEdit, QGroupBox, QHBoxLayout, QFrame, QMessageBox, QStyledItemDelegate,
    QSizePolicy, QGridLayout, QDialog, QFileDialog, QInputDialog,
    QDialogButtonBox, QMainWindow, QLayout, QCheckBox, 
    QDoubleSpinBox, QScrollArea  
) 
from PyQt6.QtGui import QPixmap, QImage, QStandardItem, QStandardItemModel, QAction
from PyQt6.QtGui import QShortcut, QKeySequence, QIcon
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRect, QSize
from pynput import mouse
from dataclasses import dataclass
from PyQt6.QtWidgets import QStackedWidget
from utils import load_model, resource_path, get_gpu_memory, get_system_memory, build_precision_labels
import settings
from custom_widgets.AboutDialog import AboutDialog
from custom_widgets.CollapsibleGroupBox import CollapsibleGroupBox
from custom_widgets.DisclaimerDialog import DisclaimerDialog
from custom_widgets.LLMChatDialog import LLMChatDialog
from custom_widgets.ResizeImageDialog import ResizableImageDialog

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
    
    # Eccentricity 任务专用
    ecc_mean_sum: float = 0.0
    ecc_median_sum: float = 0.0
    ecc_mid50_mean_sum: float = 0.0
    ecc_std_sum: float = 0.0
    cellularity_sum: float = 0.0
    
    # Cell Profiler 任务专用 (动态字典)
    dynamic_sums: dict = None

    def __post_init__(self):
        if self.dynamic_sums is None:
            self.dynamic_sums = {}

    def reset(self):
        self.total_area_mm2 = self.conf_sum = self.mitosis = 0
        self.n_tiles = self.mib_pos = self.mib_total = 0
        
        self.ecc_mean_sum = 0.0
        self.ecc_median_sum = 0.0
        self.ecc_mid50_mean_sum = 0.0
        self.ecc_std_sum = 0.0
        self.cellularity_sum = 0.0
        self.dynamic_sums = {} # 重置时清空字典
    
# >>> [这里需要新增一个初始化函数] >>>
    def __post_init__(self):
        if self.dynamic_sums is None:
            self.dynamic_sums = {}
    # <<< [新增结束] <<<

    def reset(self):
        self.total_area_mm2 = self.conf_sum = self.mitosis = 0
        self.n_tiles = self.mib_pos = self.mib_total = 0
        
        # >>> [修改开始 2.1] 重置这 4 个变量 >>>
        self.ecc_mean_sum = 0.0
        self.ecc_median_sum = 0.0
        self.ecc_mid50_mean_sum = 0.0
        self.ecc_std_sum = 0.0
        self.cellularity_sum = 0.0
        self.dynamic_sums = {}
        # <<< [修改结束 2.1] <<<
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
            # extra_cfg = {
            #     k: w.text() for k, w in self.ui.additional_config_inputs.items()
            # }
            # # >>> [修改开始 1] 支持读取 ComboBox 的值 >>>
            # extra_cfg = {}
            # for k, w in self.ui.additional_config_inputs.items():
            #     if isinstance(w, QComboBox):
            #         extra_cfg[k] = w.currentText()
            #     else:
            #         extra_cfg[k] = w.text()
            # # <<< [修改结束 1] <<<
            # # >>> [修改开始 1] 支持读取 ComboBox 和 CheckBox 的值 >>>
            # extra_cfg = {}
            # for k, w in self.ui.additional_config_inputs.items():
            #     if isinstance(w, QComboBox):
            #         extra_cfg[k] = w.currentText()
            #     elif isinstance(w, QCheckBox):
            #         extra_cfg[k] = w.isChecked()
            #     else:
            #         extra_cfg[k] = w.text()
            # # <<< [修改结束 1] <<<
            
            # # >>> mpp[修改开始 1] 支持读取 ComboBox 和 CheckBox 的值 >>>
# >>> [修改开始 4] 提取 CheckableComboBox 的勾选值 >>>
            extra_cfg = {}
            for k, w in self.ui.additional_config_inputs.items():
                if isinstance(w, CheckableComboBox):
                    # 获取被打勾的列表，例如 ['Size.Area', 'Nucleus.Intensity.Mean']
                    extra_cfg[k] = w.get_checked_items()
                elif isinstance(w, QComboBox):
                    extra_cfg[k] = w.currentText()
                elif isinstance(w, QCheckBox):
                    extra_cfg[k] = w.isChecked()
                else:
                    extra_cfg[k] = w.text()
            # <<< [修改结束 4] <<<

            # --- 新增：检测 MPP 是否处于已校准状态 ---
            saved_mpp = self.ui.calibrated_mpp
            if saved_mpp and "mpp" in extra_cfg:
                try:
                    # 如果用户没手动去乱改输入框的值，就标记为已校准
                    if abs(float(extra_cfg["mpp"]) - saved_mpp) < 1e-5:
                        extra_cfg["is_calibrated"] = True
                    else:
                        extra_cfg["is_calibrated"] = False
                except ValueError:
                    extra_cfg["is_calibrated"] = False
            else:
                extra_cfg["is_calibrated"] = False
            # ----------------------------------------
            # <<< mpp[修改结束 1] <<<

            frame, res_txt, metrics = self.process_region(
                self.ui.selected_region,
                model=self.model,
                metadata=self.metadata,
                additional_configs=extra_cfg,
            )
            #res_txt += f"\n({time.time() - t0:.4f}s)"
            res_txt += f"<br><span style='color:gray; font-size:8pt;'>({time.time() - t0:.4f}s)</span>"
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
        self.enlarged_window = None

        # Load persistent settings
        def get_user_settings_path(filename="settings.json"):
            local_appdata = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
            settings_dir = os.path.join(local_appdata, "OnSightPathology", "Settings")
            os.makedirs(settings_dir, exist_ok=True)
            return os.path.join(settings_dir, filename)

        self.settings_file = get_user_settings_path()
        self.settings = self._load_settings()

        # # >>> 新增：加载全局校准的 MPP >>>
        # self.calibrated_mpp = self.settings.get("calibrated_mpp", None)
        # # <<< 新增结束 <<<

        # >>> 字典化管理 MPP (5x, 10x, 20x, 40x) >>>
        self.calibrated_mpps = self.settings.get("calibrated_mpps", {"5x": None, "10x": None, "20x": None, "40x": None})
        self.current_mag = self.settings.get("current_mag", "20x") # 默认档位
        self.calibrated_mpp = self.calibrated_mpps.get(self.current_mag)
        # <<< 字典化管理 MPP 结束 <<<

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

    # >>> 字典化管理 MPP (5x, 10x, 20x, 40x) >>>
    def _update_global_mpp_label(self):
        """刷新左侧显示的 MPP 数值"""
        if not hasattr(self, 'lbl_global_mpp'): return
        mpp = self.calibrated_mpps.get(self.current_mag)
        if mpp:
            self.lbl_global_mpp.setText(f"MPP: {mpp:.3f} μm/px")
            self.lbl_global_mpp.setStyleSheet("color: #1abc9c; font-weight: bold;")
        else:
            self.lbl_global_mpp.setText("Not calibrated")
            self.lbl_global_mpp.setStyleSheet("color: grey;")

    def _on_global_mag_changed(self, text):
        """当用户在左侧切换 5x/10x/20x/40x 时触发"""
        self.current_mag = text
        self.settings["current_mag"] = text
        self.calibrated_mpp = self.calibrated_mpps.get(text)
        self._save_settings()
        self._update_global_mpp_label()
        
        # 同步更新 Additional Configs 里的 mpp 输入框
        if "mpp" in self.additional_config_inputs:
            if self.calibrated_mpp is not None:
                self.additional_config_inputs["mpp"].setText(f"{self.calibrated_mpp:.3f}")
            else:
                # 如果这个倍数还没被校准过，退回模型的默认预设值
                meta = settings.MODEL_METADATA.get(self.cmb_model.currentText(), {})
                self.additional_config_inputs["mpp"].setText(str(meta.get("mpp", 0.25)))
        # <<< 字典化管理 MPP 结束 <<<
        # ==================== Magnification 联动逻辑 ====================
# ==================== Magnification 联动逻辑 ====================
    # ==================== Magnification 联动逻辑 ====================
    def _on_auto_mag_clicked(self, checked):
        """当用户手动点击 Auto-sync AI 勾选框时触发"""
        
        # 🚀 [拦截器] 如果用户想打勾，先查岗：探测器开了吗？
        if checked:
            if not hasattr(self, 'mag_widget') or not self.mag_widget.is_tracking:
                # 没开！立刻弹窗教育用户
                QMessageBox.warning(
                    self, 
                    "Detector Not Running", 
                    "Please start the 'Real-time Mag' detector first before enabling Auto-sync."
                )
                # 悄悄把勾选框退回到“未勾选”状态，并且不触发信号避免死循环
                self.chk_auto_mag.blockSignals(True)
                self.chk_auto_mag.setChecked(False)
                self.chk_auto_mag.blockSignals(False)
                return # 终止后续操作，不锁下拉菜单
                
        # 如果查岗通过，或者用户是主动取消勾选，正常执行：
        self.settings["auto_mag_sync"] = checked
        self._save_settings()
        
        # 开启自动同步时，将手动下拉菜单变灰锁定；关闭时解锁
        if hasattr(self, 'cmb_global_mag'):
            self.cmb_global_mag.setEnabled(not checked)

    # def _on_mag_tracking_changed(self, is_tracking):
    #     """当探测器被用户手动关闭时，自动扯掉 Auto-sync 的勾"""
    #     if not is_tracking and hasattr(self, 'chk_auto_mag'):
    #         if self.chk_auto_mag.isChecked():
    #             self.chk_auto_mag.blockSignals(True)
    #             self.chk_auto_mag.setChecked(False)
    #             self.chk_auto_mag.blockSignals(False)
                
    #             self.settings["auto_mag_sync"] = False
    #             self._save_settings()
                
    #             if hasattr(self, 'cmb_global_mag'):
    #                 self.cmb_global_mag.setEnabled(True)

    # def _on_mag_tracking_changed(self, is_tracking):
    #     """核心状态机：探测器开启时解锁选项并恢复设置，关闭时灰掉选项"""
    #     if hasattr(self, 'chk_auto_mag'):
    #         self.chk_auto_mag.setEnabled(is_tracking)
            
    #         if is_tracking:
    #             self.chk_auto_mag.setToolTip("Automatically sync AI detected magnification")
    #             # 探测器启动了，恢复用户上次的 Auto-sync 偏好
    #             was_auto = self.settings.get("auto_mag_sync", False)
    #             self.chk_auto_mag.setChecked(was_auto)
    #             if hasattr(self, 'cmb_global_mag'):
    #                 self.cmb_global_mag.setEnabled(not was_auto)
    #         else:
    #             self.chk_auto_mag.setToolTip("Please start the Real-time Mag detector first.")
    #             self.chk_auto_mag.setChecked(False)
    #             if hasattr(self, 'cmb_global_mag'):
    #                 self.cmb_global_mag.setEnabled(True)
    def _on_mag_tracking_changed(self, is_tracking):
        """核心状态机：探测器开启时解锁选项并恢复设置，关闭时灰掉选项"""
        if hasattr(self, 'chk_auto_mag'):
            self.chk_auto_mag.setEnabled(is_tracking)
            
            if is_tracking:
                self.chk_auto_mag.setToolTip("Automatically sync AI detected magnification")
                # 探测器启动了，恢复用户上次的 Auto-sync 偏好
                was_auto = self.settings.get("auto_mag_sync", False)
                self.chk_auto_mag.setChecked(was_auto)
                if hasattr(self, 'cmb_global_mag'):
                    self.cmb_global_mag.setEnabled(not was_auto)
            else:
                self.chk_auto_mag.setToolTip("Please start the Real-time Mag detector first.")
                self.chk_auto_mag.setChecked(False)
                if hasattr(self, 'cmb_global_mag'):
                    self.cmb_global_mag.setEnabled(True)
                    
        # >>> [新增] 探测器关闭时，隐藏红色警告框 >>>
        if not is_tracking and hasattr(self, 'lbl_mag_warning'):
            self.lbl_mag_warning.hide()
        # <<< [新增结束] <<<
    # def _on_mag_detected(self, continuous_mag, pred_cls):
    #         """当 AI 检测到新倍数时，自动切档"""
    #         # 直接读取 UI 勾选框的真实状态最稳妥
    #         if not self.chk_auto_mag.isChecked() or not (hasattr(self, 'mag_widget') and self.mag_widget.is_tracking):
    #             return
                
    #         if hasattr(self, 'cmb_global_mag'):
    #             import re
    #             match = re.search(r'\d+', str(pred_cls))
    #             if match:
    #                 clean_cls = f"{match.group()}x"
    #             else:
    #                 clean_cls = str(pred_cls).lower()
                
    #             # 只有发现倍数真变了，才操作 UI
    #             if self.cmb_global_mag.currentText() != clean_cls:
    #                 index = self.cmb_global_mag.findText(clean_cls)
    #                 if index >= 0:
    #                     self.cmb_global_mag.setCurrentIndex(index)
    def _on_mag_detected(self, continuous_mag, pred_cls):
            """当 AI 检测到新倍数时，自动切档并进行警告提示"""
            
            # --- 1. 处理倍数文本清洗 ---
            import re
            match = re.search(r'\d+', str(pred_cls))
            if match:
                clean_cls = f"{match.group()}x"
            else:
                clean_cls = str(pred_cls).lower()

# >>> [修改开始] 倍数不匹配警告逻辑（支持多倍数） >>>
            if hasattr(self, 'supported_mags') and hasattr(self, 'lbl_mag_warning'):
                # 只要探测器开着，就实时比对当前倍数是否在支持的列表里
                if hasattr(self, 'mag_widget') and self.mag_widget.is_tracking:
                    # 🚀 判断当前倍数是否在支持的列表里
                    if clean_cls not in self.supported_mags:
                        req_str = " or ".join(self.supported_mags)
                        # 🚀 使用 HTML 富文本让排版更现代、重点更突出
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
            # <<< [修改结束] <<<

            # --- 2. 原有的 Auto-sync AI 联动逻辑 ---
            if not self.chk_auto_mag.isChecked() or not (hasattr(self, 'mag_widget') and self.mag_widget.is_tracking):
                return
                
            if hasattr(self, 'cmb_global_mag'):
                # 只有发现倍数真变了，才操作 UI
                if self.cmb_global_mag.currentText() != clean_cls:
                    index = self.cmb_global_mag.findText(clean_cls)
                    if index >= 0:
                        self.cmb_global_mag.setCurrentIndex(index)
    # ================================================================
    # ================================================================
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

    #############################################################################
    #clustering新代码
    #############################################################################
    def _update_overlay_params(self):
            """Pushes new spinbox values to the ACTIVE overlay widget in real-time."""
            if hasattr(self, 'current_active_overlay') and self.current_active_overlay:
                method = self.cmb_roi_method.currentText()
                
                # 🚀 [新增参数传递]
                if method == "Cluster":
                    self.current_active_overlay.patch_size = self.spin_patch.value()
                    self.current_active_overlay.n_clusters = self.spin_clusters.value()
                    self.current_active_overlay.sat_thresh = self.spin_sat.value()
                    self.current_active_overlay.val_thresh = self.spin_val.value()
                    self.current_active_overlay.tissue_thresh = self.spin_tissue.value()
                elif method == "Hotspot":
                    self.current_active_overlay.percentile = self.spin_percentile.value()
                    self.current_active_overlay.kernel_size = self.spin_kernel.value()
    #############################################################################
    #clustering新代码
    #############################################################################

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
        # >>> [新增/修改代码] 弹出对话框让用户输入自定义名字 >>>
        default_name = "Aggregate_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        custom_name, ok = QInputDialog.getText(
            self, "Export Folder Name",
            "Please enter a name for the export folder:",
            QLineEdit.EchoMode.Normal, default_name
        )
        
        # 如果用户点了取消，或者把名字删光了没填，就直接退出
        if not ok or not custom_name.strip():
            return
            
        base_name = custom_name.strip()
        # <<< [修改结束] <<<
        # ── Automatic generate subfolder ─────────────────────────────────
        #base_name = "Agregate_" + datetime.now().strftime("%Y%m%d_%H%M%S")
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

        # >>> [新增代码] 为 Eccentricity 任务在最后加上 Average 行 >>>
        meta = settings.MODEL_METADATA[self.cmb_model.currentText()]
        agg_type = meta.get("aggregate_type", "classification")
        
        if agg_type == "eccentricity" and rows:
            avg_row = {k: "" for k in rows[0].keys()}
            avg_row["file_orig"] = "OVERALL AVERAGE" 
            
            # 一个小辅助函数，用来算当前列的平均值
            def get_col_avg(col_name):
                vals = [r[col_name] for r in rows if col_name in r and isinstance(r[col_name], (int, float))]
                return sum(vals) / len(vals) if vals else ""

            avg_row["Cellularity (%)"] = get_col_avg("Cellularity (%)")
            avg_row["Ecc Mean"] = get_col_avg("Ecc Mean")
            avg_row["Ecc Median"] = get_col_avg("Ecc Median")
            avg_row["Ecc Mid-50% Mean"] = get_col_avg("Ecc Mid-50% Mean")
            avg_row["Ecc Std Dev"] = get_col_avg("Ecc Std Dev")
                
            rows.append(avg_row)
        elif agg_type == "cell_profiler" and rows:
            avg_row = {k: "" for k in rows[0].keys()}
            avg_row["file_orig"] = "OVERALL AVERAGE" 
            
            # Dynamically calculate the average for ANY column that contains numbers
            for col_name in rows[0].keys():
                if col_name not in ["file_orig", "file_annotated", "Class"]:
                    vals = [r[col_name] for r in rows if col_name in r and isinstance(r[col_name], (int, float))]
                    avg_row[col_name] = sum(vals) / len(vals) if vals else ""
                
            rows.append(avg_row)
        # <<< [新增结束] <<<

        # ── wrte CSV ─────────────────────────────────────────────
        csv_path = os.path.join(out_dir, "aggregate_metrics.csv")
        with open(csv_path, "w", newline="", encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        # ── Complete ───────────────────────────────────────────
        QMessageBox.information(
            self, "Export complete",
            f"Saved {len(rows)-1} images and metrics to:\n{out_dir}")

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
           # mm2 = m["area_px"] * (m["mpp"] / 1000) ** 2
            current_mpp = self.calibrated_mpp if self.calibrated_mpp is not None else m.get("mpp", 0.25)
            mm2 = m["area_px"] * (current_mpp / 1000.0) ** 2
        # >>> [修复1] 必须在这里初始化，否则跑其他模型会报错找不到变量 >>>
        add_ecc_mean = 0.0
        add_ecc_median = 0.0
        add_ecc_mid50_mean = 0.0
        add_ecc_std = 0.0
        add_cell = 0.0
        # <<< [修复结束] <<<

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

        # >>> [新增] Eccentricity Task 分支 >>>
# ---------- 1. 传统的 Eccentricity 任务 ----------
        elif agg_type == "eccentricity":
            # 使用 .get() 防崩溃。如果传进来的参数名字变了，这里会自动变成 0，不会闪退
            cell_val = m.get("cellularity", 0.0)
            ecc_mean = m.get("ecc_mean", 0.0)
            ecc_median = m.get("ecc_median", 0.0)
            ecc_mid50_mean = m.get("ecc_mid50_mean", 0.0)
            ecc_std = m.get("ecc_std", 0.0)

            reply = QMessageBox.question(
                self, "Add this tile?",
                (f"Cellularity: {cell_val * 100:.2f}%\n"
                 f"Mean: {ecc_mean:.4f} | Median: {ecc_median:.4f}\n"
                 f"Mid-50% Mean: {ecc_mid50_mean:.4f} | Std: {ecc_std:.4f}\n"
                 f"Tile area: {mm2:.3f} mm²\n\n"
                 "Add to aggregate?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes: return
            
            add_conf = add_mitos = add_pos = add_tot = 0
            # 提取给下面累加器用的变量
            add_ecc_mean = ecc_mean
            add_ecc_median = ecc_median
            add_ecc_mid50_mean = ecc_mid50_mean
            add_ecc_std = ecc_std
            add_cell = cell_val
            
            new_metrics = {
                "Cellularity (%)": cell_val * 100,
                "Ecc Mean": ecc_mean,
                "Ecc Median": ecc_median,
                "Ecc Mid-50% Mean": ecc_mid50_mean,
                "Ecc Std Dev": ecc_std,
                "Tissue area (mm2)": mm2
            }

        # ---------- 2. 全新的 Cell Profiler 任务 (动态收集) ----------
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
            add_ecc_mean = add_ecc_median = add_ecc_mid50_mean = add_ecc_std = 0.0
            
            # 动态抓取 metrics 里的所有数字特征
            new_metrics = {"Tissue area (mm2)": mm2}
            for key, value in m.items():
                if isinstance(value, (int, float)):
                    new_metrics[key] = value
                    # 后台隐式累加，不会把界面撑爆
                    self.agg.dynamic_sums[key] = self.agg.dynamic_sums.get(key, 0.0) + value
        # <<< [Modified] <<<
        
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
        
    # >>> [新增] 累加 eccentricity 数据 >>>
        self.agg.ecc_mean_sum += add_ecc_mean
        self.agg.ecc_median_sum += add_ecc_median
        self.agg.ecc_mid50_mean_sum += add_ecc_mid50_mean
        self.agg.ecc_std_sum += add_ecc_std
        self.agg.cellularity_sum += add_cell
        # <<< [新增结束] <<<
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

        # >>> [新增] Eccentricity 显示逻辑 >>>
# ---------- 1. 传统的 Eccentricity 界面显示 ----------
        elif agg_type == "eccentricity":
            n = self.agg.n_tiles
            avg_ecc_mean = (self.agg.ecc_mean_sum / n) if n else 0
            avg_ecc_median = (self.agg.ecc_median_sum / n) if n else 0
            avg_ecc_mid50_mean = (self.agg.ecc_mid50_mean_sum / n) if n else 0
            avg_ecc_std = (self.agg.ecc_std_sum / n) if n else 0
            avg_cell = (self.agg.cellularity_sum / n) if n else 0
            
            lines += [
                f"Avg Cellularity: {avg_cell * 100:.2f}%",
                f"Avg Ecc Mean: {avg_ecc_mean:.4f}",
                f"Avg Ecc Median: {avg_ecc_median:.4f}",
                f"Avg Ecc Mid-50%: {avg_ecc_mid50_mean:.4f}",
                f"Avg Ecc Std Dev: {avg_ecc_std:.4f}",
            ]
            
        # ---------- 2. 全新的 Cell Profiler 界面显示 ----------
        elif agg_type == "cell_profiler":
            n = self.agg.n_tiles
            # 从动态字典里安全读取你最想展示的几个核心指标
            avg_cell = self.agg.dynamic_sums.get('cellularity_percent', 0) / n if n else 0
            avg_area = self.agg.dynamic_sums.get('Size.Area_mean', 0) / n if n else 0
            avg_circ = self.agg.dynamic_sums.get('Shape.Circularity_mean', 0) / n if n else 0
            avg_ecc = self.agg.dynamic_sums.get('Shape.Eccentricity_mean', 0) / n if n else 0
            median_ecc = self.agg.dynamic_sums.get('Shape.Eccentricity_median', 0) / n if n else 0

            
            
            lines += [
                f"Avg Cellularity: {avg_cell:.2f} cells/mm²",
                #f"Avg Nuclear Area: {avg_area:.2f} μm²",
                #f"Avg Circularity: {avg_circ:.4f}",
                f"Avg Eccentricity: {avg_ecc:.4f}",
                f"Median Eccentricity: {median_ecc:.4f}",
                "",
                f"✓ Tracking {len(self.agg.dynamic_sums)} total features",
                f"  (Will be saved on Export)"
            ]
        # <<< [新增结束] <<<
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

        # ───────── Second Click (Custom Dialog) ─────────
        dialog = QDialog(self)
        dialog.setWindowTitle("Calibrate Area")
        form = QVBoxLayout(dialog)
        form.addWidget(QLabel("Select magnification profile and enter area:"))
        
        h_layout = QHBoxLayout()
        
        # --- 左侧：Magnification 下拉菜单 ---
        cmb_mag = QComboBox()
        cmb_mag.addItems(["5x", "10x", "20x", "40x"])
        if hasattr(self, 'cmb_global_mag'):
            cmb_mag.setCurrentText(self.cmb_global_mag.currentText()) # 默认选中主界面正在用的倍数
            
        # --- 右侧：SpinBox 和 单位 ---
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

        # 核心：自动聚焦并全选，让用户打字直接覆盖！
        spin_val.setFocus()
        spin_val.selectAll()

        if dialog.exec() == QDialog.DialogCode.Accepted and spin_val.value() > 0:
            val = spin_val.value()
            mag_key = cmb_mag.currentText()
            
            # 统一转换为平方微米 (um²) 来计算 MPP
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
                
                # 计算出真实的 MPP: sqrt( 物理面积 / 像素面积 )
                new_mpp = math.sqrt(val_um2 / (physical_box_w * physical_box_h))
                
                # 保存到字典中并持久化
                self.calibrated_mpps[mag_key] = new_mpp
                self.settings["calibrated_mpps"] = self.calibrated_mpps
                self._save_settings()
                
                # --- 新增：强制 UI 刷新逻辑 ---
                if hasattr(self, 'cmb_global_mag'):
                    if self.cmb_global_mag.currentText() == mag_key:
                        # 如果主界面当前正停留在被校准的倍数上，手动强制刷新 UI
                        self.calibrated_mpp = new_mpp
                        self._update_global_mpp_label()
                        if "mpp" in self.additional_config_inputs:
                            self.additional_config_inputs["mpp"].setText(f"{new_mpp:.3f}")
                    else:
                        # 如果校准的是其他倍数，直接切过去（PyQt 会自动触发切换信号并刷新）
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
    # def _calibrate_area(self):
    #     # ───────── First Click ─────────
    #     if not self.show_calib_box:
    #         if not self._calib_help_shown:
    #             QMessageBox.information(
    #                 self, "How to calibrate",
    #                 "Step 1: A translucent box will appear.\n"
    #                 "Step 2: In your slide viewer draw a box that exactly overlaps it\n"
    #                 "    and note the physical area.\n"
    #                 "Step 3: Click the 'Enter Area' button and input the measured value.")
    #             self._calib_help_shown = True

    #         self.show_calib_box = True
    #         QApplication.processEvents()
    #         self.btn_calib.setText("Enter area")
    #         return

    #     # ───────── Second Click (Custom Dialog) ─────────
    #     dialog = QDialog(self)
    #     dialog.setWindowTitle("Calibrate Area")
    #     form = QVBoxLayout(dialog)
    #     form.addWidget(QLabel("Enter the physical area of the highlighted box:"))
        
    #     h_layout = QHBoxLayout()
    #     # 使用 QDoubleSpinBox 替代旧的 QInputDialog
    #     from PyQt6.QtWidgets import QDoubleSpinBox
    #     spin_val = QDoubleSpinBox()
    #     spin_val.setDecimals(4)
    #     spin_val.setMaximum(9999999.0)
    #     spin_val.setMinimum(0.0001)
        
    #     cmb_unit = QComboBox()
    #     cmb_unit.addItems(["μm²", "mm²"])
        
    #     h_layout.addWidget(spin_val)
    #     h_layout.addWidget(cmb_unit)
    #     form.addLayout(h_layout)
        
    #     btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    #     btn_box.accepted.connect(dialog.accept)
    #     btn_box.rejected.connect(dialog.reject)
    #     form.addWidget(btn_box)

    #     # 核心：自动聚焦并全选，让你打字直接覆盖！
    #     spin_val.setFocus()
    #     spin_val.selectAll()

    #     if dialog.exec() == QDialog.DialogCode.Accepted and spin_val.value() > 0:
    #         val = spin_val.value()
            
    #         # 统一转换为平方微米 (um²) 来计算 MPP
    #         if cmb_unit.currentText() == "mm²":
    #             val_um2 = val * 1_000_000.0
    #         else:
    #             val_um2 = val

    #         if self.latest_frame is not None:
    #             h, w, c = self.latest_frame.shape
    #             bx = int(w / 3)
    #             by = int(h / 3)
                
    #             # 计算出真实的 MPP: sqrt( 物理面积 / 像素面积 )
    #             new_mpp = math.sqrt(val_um2 / (bx * by))
                
    #             # 全局记录 MPP 并持久化
    #             self.calibrated_mpp = new_mpp
    #             self.settings["calibrated_mpp"] = new_mpp
    #             self._save_settings()
                
    #             # 立即同步到左侧界面的 mpp 输入框中
    #             if "mpp" in self.additional_config_inputs:
    #                 self.additional_config_inputs["mpp"].setText(f"{new_mpp:.5f}")

    #             QMessageBox.information(
    #                 self, "Calibration successful",
    #                 f"Calculated Global MPP = {new_mpp:.5f} μm/px\n"
    #                 "This MPP has been saved and will override all default settings."
    #             )
    #     else:
    #         QMessageBox.information(self, "Cancelled", "No valid area recorded.")

    #     # Close and Reset
    #     self.show_calib_box = False
    #     self.btn_calib.setText("Calibrate")
    # # ------------------------------------------------------------
    # # CALIBRATION DIALOG 
    # # ------------------------------------------------------------
    # def _calibrate_area(self):
    #     # ───────── First Click ─────────
    #     if not self.show_calib_box:
    #         # First Time Instruction Pop-up
    #         if not self._calib_help_shown:
    #             QMessageBox.information(
    #                 self, "How to calibrate",
    #                 "Step 1: A translucent box will appear.\n"
    #                 "Step 2: In your slide viewer draw a box that exactly overlaps it\n"
    #                 "    and note the physical area (mm²).\n"
    #                 "Step 3: Click the 'Enter Area' button and input the measured value.")
    #             self._calib_help_shown = True

    #         # Open Calibrate Box
    #         self.show_calib_box = True
    #         QApplication.processEvents()

    #         # Wait for Second Click
    #         self.btn_calib.setText("Enter area")
    #         return

    #     # ───────── Second Click ─────────
    #     val, ok = QInputDialog.getDouble(
    #         self, "Calibrate Area",
    #         "Enter the measured area of the highlighted box (mm²):",
    #         decimals=4, min=0.0001
    #     )

    #     if ok and val > 0:
    #         self.box_mm2 = val
    #     # >>> mpp[修改开始 3] 根据用户输入的面积自动计算并保存 MPP >>>
    #         if self.latest_frame is not None:
    #             h, w, c = self.latest_frame.shape
    #             bx = int(w / 3)
    #             by = int(h / 3)
                
    #             # 计算像素真实代表的面积，开根号得出 MPP
    #             pixel_area_um2 = (val * 1000000.0) / (bx * by)
    #             calibrated_mpp = math.sqrt(pixel_area_um2)
                
    #             # 永久保存到本地设置
    #             self.settings["calibrated_mpp"] = calibrated_mpp
    #             self._save_settings()

    #             # 如果左侧配置栏有 mpp 输入框，立刻更新显示
    #             if "mpp" in self.additional_config_inputs:
    #                 self.additional_config_inputs["mpp"].setText(f"{calibrated_mpp:.5f}")

    #             QMessageBox.information(
    #                 self, "Calibration stored",
    #                 f"Calibration box = {self.box_mm2:.4f} mm²\n"
    #                 f"Calculated MPP = {calibrated_mpp:.5f} μm/px\n\n"
    #                 "This MPP has been saved and will be used automatically."
    #             )
    #         # <<< mpp[修改结束 3] <<<
    #     else:
    #         QMessageBox.information(
    #             self, "Calibration cancelled",
    #             "No calibration value recorded.")

    #     # Close and Reset
    #     self.show_calib_box = False
    #     self.btn_calib.setText("Calibrate")

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
        # +++ 插入 Real-time Mag Detector Widget +++
        self.mag_widget = MagDetectorWidget(get_region_callback=lambda: self.selected_region)
        self.mag_widget.mag_detected.connect(self._on_mag_detected)
        self.mag_widget.tracking_state_changed.connect(self._on_mag_tracking_changed)
        v_cap.addWidget(self.mag_widget)
        # +++++++++++++++++++++++++++++++++++++++++
        
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



        # >>> 字典化管理 MPP (5x, 10x, 20x, 40x) >>>
        grp_mag = QGroupBox("Active Magnification")
        h_mag = QHBoxLayout()
        
        self.cmb_global_mag = QComboBox()
        self.cmb_global_mag.addItems(["5x", "10x", "20x", "40x"])
        self.cmb_global_mag.setCurrentText(self.current_mag)
        self.cmb_global_mag.currentTextChanged.connect(self._on_global_mag_changed)
        
        self.lbl_global_mpp = QLabel()
        self._update_global_mpp_label()

        # >>> [新增] Auto-sync AI 联动选项 >>>
# >>> [修改] Auto-sync AI 状态机优化 >>>
# >>> [修改] Auto-sync UI 初始化 >>>
        self.chk_auto_mag = QCheckBox("Auto-sync AI")
        self.chk_auto_mag.setStyleSheet("color: #3498db; font-weight: bold;")
        # 软件刚启动时探测器肯定没开，所以强制默认为 False
        self.chk_auto_mag.setChecked(False) 
        self.settings["auto_mag_sync"] = False 
        
        self.chk_auto_mag.clicked.connect(self._on_auto_mag_clicked)
        # <<< [修改] <<<
        # <<< [修改] <<<
        # <<< [新增] <<<

        h_mag.addWidget(self.cmb_global_mag)
        h_mag.addWidget(self.chk_auto_mag)  # 将复选框加进去
        h_mag.addWidget(self.lbl_global_mpp)
        grp_mag.setLayout(h_mag)
        main_left.addWidget(grp_mag)
        # <<< 字典化管理 MPP 结束 <<<

        # -------------------- Additional configs --------------------------
        self.grp_cfg = QGroupBox("Additional Configs")
        self.v_cfg = QVBoxLayout()
        self.grp_cfg.setLayout(self.v_cfg)
        main_left.addWidget(self.grp_cfg)

        # =========================================================================
       # =========================================================================
        # >>> [修改] 将 ROI Finder 的控制面板移到左侧，并将下拉菜单和参数放在同一行 >>>
        grp_roi = QGroupBox("ROI Finder")
        v_roi = QVBoxLayout()

        h_roi_top = QHBoxLayout()
        h_roi_top.addWidget(QLabel("Method:"))
        self.cmb_roi_method = QComboBox()
        # 🚀 [顺序调换] 将 Hotspot 放在前面
        self.cmb_roi_method.addItems(["Hotspot", "Cluster"]) 
        self.cmb_roi_method.currentIndexChanged.connect(self._on_roi_method_changed)
        h_roi_top.addWidget(self.cmb_roi_method)
        h_roi_top.addSpacing(10)

        self.stacked_params = QStackedWidget()
        
        # --- 卡片 1：Hotspot 参数页 (移到了前面) ---
        page_hotspot = QWidget()
        # 初始默认显示 Hotspot，设置它的尺寸策略为 Preferred
        page_hotspot.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        h_hotspot = QHBoxLayout(page_hotspot)
        h_hotspot.setContentsMargins(0, 0, 0, 0)
        
        self.lbl_percentile = QLabel("Percentile:")
        self.spin_percentile = QSpinBox()
        self.spin_percentile.setRange(10, 99)
        self.spin_percentile.setValue(60)
        self.spin_percentile.setSingleStep(5)
        self.spin_percentile.valueChanged.connect(self._update_overlay_params)

        
        # 🚀 [新增] Kernel 参数框
        self.lbl_kernel = QLabel("Kernel:")
        self.spin_kernel = QSpinBox()
        self.spin_kernel.setRange(1, 99) 
        self.spin_kernel.setValue(5)      # 默认值 5
        self.spin_kernel.setSingleStep(2) # 建议奇数步进 (3, 5, 7...)
        self.spin_kernel.valueChanged.connect(self._update_overlay_params)


        h_hotspot.addWidget(self.lbl_percentile)
        h_hotspot.addWidget(self.spin_percentile)
        h_hotspot.addWidget(self.lbl_kernel)
        h_hotspot.addWidget(self.spin_kernel)
        h_hotspot.addStretch() 
        self.stacked_params.addWidget(page_hotspot) # 索引 0

        # --- 卡片 2：Cluster 参数页 (🚀 完美对齐网格版) ---
        page_cluster = QWidget()
        # 初始时它是隐藏的，必须设为 Ignored 才能消除幽灵空白
        page_cluster.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored) 
        
        grid_cluster = QGridLayout(page_cluster)
        grid_cluster.setContentsMargins(0, 0, 0, 0)
        grid_cluster.setSpacing(8) # 让上下两排紧凑一点
        
        # 第一排 (Row 0)
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

        # 第二排 (Row 1)
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
        
        # 🚀 在最右侧（第 6 列）加一个弹簧，把所有格子紧紧挤在左边对齐
        grid_cluster.setColumnStretch(6, 1) 

        self.stacked_params.addWidget(page_cluster) # 索引 1
        # -------------------------------------------------------------

        h_roi_top.addWidget(self.stacked_params)
        v_roi.addLayout(h_roi_top)

        # 2. 启停按钮 (单独放下一行)
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


        # <<< [新增结束] <<<
        # =========================================================================
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
 # -------------------- Display output -----------------------------
        grp_out = QGroupBox("Inference Output")
        v_out = QVBoxLayout()

        # 1. 必须先创建底图标签 lbl_img
        self.lbl_img = QLabel()
        self.lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 2. 再把两个透明图层 (Overlay) 像贴膜一样盖在 lbl_img 上
        lbl_layout = QVBoxLayout(self.lbl_img)
        lbl_layout.setContentsMargins(0, 0, 0, 0)
        
        self.overlay_clustering = ClusteringOverlay(self.lbl_img)
        self.overlay_clustering.hide() 
        
        self.overlay_histomics = HistomicsOverlay(self.lbl_img)
        self.overlay_histomics.hide() 
        
        lbl_layout.addWidget(self.overlay_clustering, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        lbl_layout.addWidget(self.overlay_histomics, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        # >>> [新增] 动态倍数警告标签，放在图片左下角 >>>
        self.lbl_mag_warning = QLabel("⚠️ Warning: Magnification mismatch!")
        self.lbl_mag_warning.setStyleSheet("""
            background-color: rgba(30, 40, 50, 220); /* 偏深蓝灰的半透明底色，比纯黑高级 */
            padding: 8px 14px; 
            border: 1px solid #ff4c4c; /* 加一圈淡淡的红色描边，强调这是警告 */
            border-radius: 6px;
        """)
        self.lbl_mag_warning.hide() # 默认隐藏
        lbl_layout.addWidget(self.lbl_mag_warning, alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft)
        # <<< [新增结束] <<<
        # >>> [新增] 把两个图层的后台完成信号接入放大窗口 >>>
        self.overlay_clustering.worker.finished.connect(self._on_overlay_worker_finished)
        self.overlay_histomics.worker.finished.connect(self._on_overlay_worker_finished)
        # <<< [新增结束] <<<
        
        self.current_active_overlay = None 

        v_out.addWidget(self.lbl_img)
        
        # 3. 结果文字和放大按钮
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
        # -------------------------------------------------------------------------

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
        self._on_roi_method_changed(0)
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
# +++ 插入 Real-time Mag Detector Widget +++
        self.mag_widget = MagDetectorWidget(get_region_callback=lambda: self.selected_region)
        self.mag_widget.mag_detected.connect(self._on_mag_detected)
        self.mag_widget.tracking_state_changed.connect(self._on_mag_tracking_changed)
        v_cap.addWidget(self.mag_widget)
        # +++++++++++++++++++++++++++++++++++++++++
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
# -------------------- Display output (Compact 模式同步更新) -----------------------------
        grp_out = QGroupBox("Inference Output")
        v_out = QVBoxLayout()

        # 1. 创建底图
        self.lbl_img = QLabel()
        self.lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 2. 创建双图层并堆叠
        lbl_layout = QVBoxLayout(self.lbl_img)
        lbl_layout.setContentsMargins(0, 0, 0, 0)
        
        self.overlay_clustering = ClusteringOverlay(self.lbl_img)
        self.overlay_clustering.hide() 
        
        self.overlay_histomics = HistomicsOverlay(self.lbl_img)
        self.overlay_histomics.hide() 
        
        lbl_layout.addWidget(self.overlay_clustering, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        lbl_layout.addWidget(self.overlay_histomics, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        
        self.current_active_overlay = None 

        v_out.addWidget(self.lbl_img)
        
        # --- Compact 模式下的 ROI Finder ---
 # --- Compact 模式下的 ROI Finder ---
        grp_roi = QGroupBox("ROI Finder")
        v_roi = QVBoxLayout()

        h_roi_top = QHBoxLayout()
        h_roi_top.addWidget(QLabel("Method:"))
        
        self.cmb_roi_method = QComboBox()
        # 🚀 [顺序调换]
        self.cmb_roi_method.addItems(["Hotspot", "Cluster"]) 
        self.cmb_roi_method.currentIndexChanged.connect(self._on_roi_method_changed)
        h_roi_top.addWidget(self.cmb_roi_method)
        h_roi_top.addSpacing(10)

        self.stacked_params = QStackedWidget()
        
        # 卡片 1：Hotspot
# --- 卡片 1：Hotspot 参数页 ---
        page_hotspot = QWidget()
        # 初始默认显示 Hotspot，设置它的尺寸策略为 Preferred
        page_hotspot.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        h_hotspot = QHBoxLayout(page_hotspot)
        h_hotspot.setContentsMargins(0, 0, 0, 0)
        
        self.lbl_percentile = QLabel("Percentile:")
        self.spin_percentile = QSpinBox()
        self.spin_percentile.setRange(10, 99)
        self.spin_percentile.setValue(60)
        self.spin_percentile.setSingleStep(5)
        self.spin_percentile.valueChanged.connect(self._update_overlay_params)

        # 🚀 [新增] Kernel 参数框
        self.lbl_kernel = QLabel("Kernel:")
        self.spin_kernel = QSpinBox()
        self.spin_kernel.setRange(1, 99) 
        self.spin_kernel.setValue(5)      # 默认值 5
        self.spin_kernel.setSingleStep(2) # 建议奇数步进 (3, 5, 7...)
        self.spin_kernel.valueChanged.connect(self._update_overlay_params)
        
        h_hotspot.addWidget(self.lbl_percentile)
        h_hotspot.addWidget(self.spin_percentile)
        h_hotspot.addWidget(self.lbl_kernel)
        h_hotspot.addWidget(self.spin_kernel)
        h_hotspot.addStretch() 
        self.stacked_params.addWidget(page_hotspot) # 索引 0

        # --- 卡片 2：Cluster 参数页 (🚀 完美对齐网格版) ---
        page_cluster = QWidget()
        # 初始时它是隐藏的，必须设为 Ignored 才能消除幽灵空白
        page_cluster.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored) 
        
        grid_cluster = QGridLayout(page_cluster)
        grid_cluster.setContentsMargins(0, 0, 0, 0)
        grid_cluster.setSpacing(8) # 让上下两排紧凑一点
        
        # 第一排 (Row 0)
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

        # 第二排 (Row 1)
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
        
        # 🚀 在最右侧（第 6 列）加一个弹簧，把所有格子紧紧挤在左边对齐
        grid_cluster.setColumnStretch(6, 1) 

        self.stacked_params.addWidget(page_cluster) # 索引 1

        h_roi_top.addWidget(self.stacked_params)
        v_roi.addLayout(h_roi_top)

        # 启停按钮
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

        # --- 结果与放大 ---
        self.lbl_res = QLabel("Result:")
        v_out.addWidget(self.lbl_res)

        self.btn_enlarge = QPushButton("Expanded View 🔍")
        self.btn_enlarge.setEnabled(False)
        self.btn_enlarge.clicked.connect(self._open_enlarged_view)
        v_out.addWidget(self.btn_enlarge)

        grp_out.setLayout(v_out)
        main_layout.addWidget(grp_out)

        main_layout.addStretch()
        
        # 强制初始化 ROI UI 状态
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
# --------------------------------- Overlay Controls ---------------------------------
#####overlay新代码#####
# --------------------------------- Overlay Controls ---------------------------------
    def _start_overlay(self):
        # [修复] 使用新的统一指针 current_active_overlay
        if hasattr(self, 'current_active_overlay') and self.current_active_overlay:
            
            # 启动前，确保把当前 UI 上的参数喂给对应的后台
            method = self.cmb_roi_method.currentText()
            if method == "Cluster":
                self.current_active_overlay.patch_size = self.spin_patch.value()
                self.current_active_overlay.n_clusters = self.spin_clusters.value()
            elif method == "Hotspot":
                self.current_active_overlay.percentile = self.spin_percentile.value()

            self.current_active_overlay.start()
            # >>> [绝妙修复] 如果当前主程序是停止状态（静态图），强制喂一次图给 Overlay！
            if (not self.thread or not self.thread.isRunning()) and self.latest_frame is not None:
                self.current_active_overlay.process_frame(self.latest_frame)
            # <<< [修复结束]
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
                
    # >>> [关键新增] 停止时，清空小图缓存，并告诉放大窗口隐藏
                self.latest_overlay_result = None 
                if getattr(self, 'enlarged_window', None) and self.enlarged_window.isVisible():
                    self.enlarged_window.hide_overlay()
                # <<< [新增结束]
    # ------------------------------------------------------------------------------------
    #####overlay新代码#####
    # ------------------------------------------------------------------------------------

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

        # >>> overlay新代码[修改开始 2] Show the overlay when start is clicked >>>
        if hasattr(self, 'btn_overlay_start'):
            self.btn_overlay_start.setEnabled(True)
        # <<< overlay新代码[修改结束 2] <<<

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_chat.setEnabled(True)

        self.btn_agg_start.setEnabled(True)
        self.btn_calib.setEnabled(True)

    def _stop(self):
        if self.thread:
            self.thread.stop()
            self.thread = None

        # >>> overlay新代码[修改开始 3] Hide and reset the overlay when stopped >>>
        if hasattr(self, 'btn_overlay_start'):
            self.btn_overlay_start.setEnabled(False) # Turn off until started again
        # <<< overlay新代码[修改结束 3] <<<
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
        # >>> [ATTENTION MAP OVERLAY START] >>>
        # Find the config UI element dynamically
        attn_toggle = self.additional_config_inputs.get("Show ViT Attention Map")
        
        # If it exists, is a checkbox, and is checked, apply the heatmap
        if attn_toggle and hasattr(attn_toggle, 'isChecked') and attn_toggle.isChecked():
            if "attention_map" in metrics:
                attn_map = metrics["attention_map"]
                heatmap = cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_JET)
                frame_bgr = cv2.addWeighted(heatmap, 0.4, frame_bgr, 0.6, 0)
        # <<< [ATTENTION MAP OVERLAY END] <<<

        # CENTRAL 1/3 × 1/3  CALIBRATION BOX
        if self.show_calib_box:
            bx = int(w / 3);4
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
        #self.lbl_res.setText(f"Result: {txt}")
        ############################################
        # 修改新添加代码
        ############################################
        self.lbl_res.setTextFormat(Qt.TextFormat.RichText)  # 强制设置为 HTML 富文本渲染
        self.lbl_res.setText(txt)  # 传入刚才生成的 HTML 代码
        ############################################
        ############################################


        
        self.latest_frame = frame_rgb.copy()

        # >>> overlay新代码[修改开始 3.3] 喂给右上角小窗当前RGB画面 >>>
# >>> [修复] 喂给当前激活的右上角小窗 RGB 画面 >>>
        if hasattr(self, 'current_active_overlay') and self.current_active_overlay:
            self.current_active_overlay.process_frame(self.latest_frame)
        # <<< [修复结束] <<<
        # <<< overlay新代码[修改结束 3.3] <<<

        self.last_result = txt
        self.btn_export.setEnabled(True)

        # >>> [新增代码] 如果放大窗口开着，就实时更新它的画面 >>>
        if hasattr(self, 'btn_enlarge'):
            self.btn_enlarge.setEnabled(True)
        if getattr(self, 'enlarged_window', None) and self.enlarged_window.isVisible():
            self.enlarged_window.set_image(self.latest_frame)
            if hasattr(self.enlarged_window, 'set_text'):
                self.enlarged_window.set_text(txt)
        # <<< [新增结束] <<<
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

# >>> [修改代码] 动态打开放大弹窗 >>>
    def _open_enlarged_view(self):
        if self.latest_frame is None:
            return
        
        # 如果窗口还没创建或者已经被关闭了，就新建一个
        if self.enlarged_window is None or not self.enlarged_window.isVisible():
            self.enlarged_window = ResizableImageDialog(None) 
            self.enlarged_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose) 
            self.enlarged_window.destroyed.connect(self._on_enlarged_window_closed)
            self.enlarged_window.show() 
            
        # 传递当前帧并置顶窗口
        self.enlarged_window.set_image(self.latest_frame)
        if hasattr(self.enlarged_window, 'set_text'):
            self.enlarged_window.set_text(self.last_result)
            
        # >>> [关键新增] 如果当前主界面正开着 Overlay，且我们有缓存，立刻发给放大窗口！
        if hasattr(self, 'current_active_overlay') and self.current_active_overlay and self.current_active_overlay.isVisible():
            if getattr(self, 'latest_overlay_result', None) is not None:
                self.enlarged_window.set_overlay(self.latest_overlay_result)
        # <<< [新增结束]
        
        self.enlarged_window.activateWindow()

    def _on_enlarged_window_closed(self):
        self.enlarged_window = None
    # <<< [修改结束] <<<
    # --------------------------------- Chat-------------------------------------------------
    def _open_chat(self, llm_name, llm_precision):
        if self.latest_frame is None:
            return

        #self._stop()  # frees GPU memory so that we can load LLM
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

    def _on_roi_method_changed(self, index):
            method = self.cmb_roi_method.currentText()
            
            # 切换方法时，先停止当前图层
            self._stop_overlay()
            
            if method == "Hotspot":
                self.current_active_overlay = self.overlay_histomics
                if hasattr(self, 'stacked_params'):
                    self.stacked_params.setCurrentIndex(0)
            else: # "Cluster"
                self.current_active_overlay = self.overlay_clustering
                if hasattr(self, 'stacked_params'):
                    self.stacked_params.setCurrentIndex(1)
                    
            # 🚀 [消除幽灵格子的核心魔法] 
            # 把当前显示的卡片尺寸撑开，把不显示的卡片尺寸彻底忽略，强迫窗口收缩！
            if hasattr(self, 'stacked_params'):
                for i in range(self.stacked_params.count()):
                    widget = self.stacked_params.widget(i)
                    if i == self.stacked_params.currentIndex():
                        widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
                    else:
                        widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
                # 重新计算并收缩外框尺寸
                self.stacked_params.adjustSize()
                
            # 强制刷新参数到当前的 Widget
            if hasattr(self, '_update_overlay_params'):
                self._update_overlay_params()

    def _on_overlay_worker_finished(self, result_rgb):
            """当后台计算好聚类/热力图时，同步刷新给放大窗口（使用高分辨率原图）"""
            # >>> [关键新增] 存下最新的一帧，留给稍后可能打开的放大窗口用
            self.latest_overlay_result = result_rgb.copy() 
            # <<< [新增结束]
            
            if getattr(self, 'enlarged_window', None) and self.enlarged_window.isVisible():
                self.enlarged_window.set_overlay(result_rgb)
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
        # >>> [修改开始] 升级为支持多倍数列表 >>>
        # 如果 JSON 里配置了 "supported_mags": ["20x", "40x"] 就用配置的，否则默认使用算出来的单个倍数
        self.supported_mags = meta.get("supported_mags", [f"{mag}x"])
        mag_display = " or ".join(self.supported_mags) # 用于 UI 显示，例如 "20x or 40x"
        
        if hasattr(self, 'lbl_mag_warning'):
            self.lbl_mag_warning.hide() # 切换模型时先隐藏警告
        # <<< [修改结束] <<<
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
        # >>> [新增代码 1]：增加一个字典，用来存储每一行的 Widget 容器，方便稍后隐藏它 >>>
        self.additional_config_rows = {} 
        # <<< [新增结束] <<<
        add_cfg = meta.get("additional_configs", {})
        self.grp_cfg.setVisible(bool(add_cfg))

        # Clear existing widgets
        clear_layout(self.v_cfg)

        for k, v in add_cfg.items():
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)  # optional
            lbl = QLabel(k)

            # #edt = QLineEdit(str(v))
            # # >>> [修改开始 2] 动态判断渲染 QLineEdit 还是 QComboBox >>>
            # if isinstance(v, list):
            #     # 如果 settings 里传的是列表，就变成下拉菜单
            #     edt = QComboBox()
            #     edt.addItems([str(item) for item in v])
            # else:
            #     edt = QLineEdit(str(v))
            # # <<< [修改结束 2] <<<
            # # >>> [修改开始 2] 动态判断渲染 QLineEdit, QComboBox, 还是 QCheckBox >>>
            # if isinstance(v, list):
            #     # 列表变下拉菜单
            #     edt = QComboBox()
            #     edt.addItems([str(item) for item in v])
            # elif isinstance(v, bool):
            #     # 布尔值变勾选框 (Toggle)
            #     edt = QCheckBox()
            #     edt.setChecked(v)
            # else:
            #     # 其它变文本框
            #     edt = QLineEdit(str(v))
            # # <<< [修改结束 2] <<<
            # >>> mpp[修改开始 2] 动态判断渲染控件并加载已保存的校准 MPP >>>
            # if isinstance(v, list):
            #     # 列表变下拉菜单
            #     edt = QComboBox()
            #     edt.addItems([str(item) for item in v])
            # elif isinstance(v, bool):
            # >>> [修改开始 2] 识别暗号并渲染打勾下拉菜单 >>>
            # >>> [修改开始 2] 识别暗号并渲染打勾下拉菜单 >>>
            if "(Multi-Select)" in k and isinstance(v, list):
                edt = CheckableComboBox()
                # 🚀 关键修改：把 checked_by_default 改为 False，启动时就是全空状态
                edt.addItems([str(item) for item in v], checked_by_default=False)
            elif isinstance(v, list):
                edt = QComboBox()
                edt.addItems([str(item) for item in v])
            # <<< [修改结束 2] <<<
            elif isinstance(v, bool):
            # <<< [修改结束 2] <<<
                # 布尔值变勾选框 (Toggle)
                edt = QCheckBox()
                edt.setChecked(v)
            else:
                # 其它变文本框，如果是 mpp 并且本地有校准记录，则优先加载校准记录
                if k.lower() == "mpp" and self.calibrated_mpp is not None:
                    edt = QLineEdit(f"{self.calibrated_mpp:.5f}")
                else:
                    edt = QLineEdit(str(v))
            # <<< mpp[修改结束 2] <<<


            row_layout.addWidget(lbl)
            row_layout.addWidget(edt)
            self.v_cfg.addWidget(row_widget)  # <-- add widget, not layout
            self.additional_config_inputs[k] = edt
            # >>> [新增代码 2]：把当前这整行(包含Label和输入框)存起来 >>>
            self.additional_config_rows[k] = row_widget
            # <<< [新增结束] <<<
# >>> [核心新增代码 3]：动态显示/隐藏 Context Scale 和 Clusters (k) 联动逻辑 >>>
        if "Clustering Method" in self.additional_config_inputs:
            cmb_clustering = self.additional_config_inputs["Clustering Method"]
            
            # 获取需要控制的行容器
            row_context_scale = self.additional_config_rows.get("Context Scale")
            row_k = self.additional_config_rows.get("Clusters (k)")
            
            # 🚀 [新增] 获取 Image Type 行容器
            row_img_type = self.additional_config_rows.get("Image Type")
            
            # 新增获取特征列表行
            row_feat_he = self.additional_config_rows.get("H&E Features (Multi-Select)")
            row_feat_muscle = self.additional_config_rows.get("Muscle Features (Multi-Select)")

            def update_clustering_ui():
                method_text = cmb_clustering.currentText()
                img_type_text = ""
                
                # 如果有 Image Type 下拉菜单，获取它的值
                if "Image Type" in self.additional_config_inputs:
                    img_type_text = self.additional_config_inputs["Image Type"].currentText()

                # 逻辑 1：显示 Context Scale
                if row_context_scale:
                    row_context_scale.setVisible(method_text in ["Single Cell Clustering", "Region Clustering"])
                
                # 逻辑 2：显示 Clusters (k)
                if row_k:
                    row_k.setVisible(method_text != "None")
                    
                # 🚀 逻辑 3 [核心修复]：只有选了 Customized Features 时，才显示 Image Type 下拉菜单！
                is_custom = (method_text == "Customized Features")
                if row_img_type:
                    row_img_type.setVisible(is_custom)
                    
                # 逻辑 4：仅在选 Customized Features 时，根据 Image Type 显示对应的特征打勾框！
                if row_feat_he:
                    row_feat_he.setVisible(is_custom and img_type_text == "H&E Cell Analysis")
                if row_feat_muscle:
                    row_feat_muscle.setVisible(is_custom and img_type_text == "Muscle Fiber Typing")
                    
                # 逻辑 5：显存释放
                if method_text not in ["Single Cell Clustering", "Region Clustering"]:
                    try:
                        from utils_clustering import clear_cluster_models_cache
                        clear_cluster_models_cache()
                    except ImportError: pass

            # 绑定信号：当 Clustering Method 或 Image Type 改变时，都触发刷新！
            cmb_clustering.currentTextChanged.connect(lambda _: update_clustering_ui())
            if "Image Type" in self.additional_config_inputs:
                self.additional_config_inputs["Image Type"].currentTextChanged.connect(lambda _: update_clustering_ui())
            
            # 初始化状态
            update_clustering_ui()
# <<< [核心新增结束] <<<
# <<< [核心新增结束] <<<
        # if "Clustering Method" in self.additional_config_inputs:
        #     cmb_clustering = self.additional_config_inputs["Clustering Method"]
            
        #     # 获取需要控制的行容器
        #     row_context_scale = self.additional_config_rows.get("Context Scale")
        #     row_k = self.additional_config_rows.get("Clusters (k)")

        #     # 定义统一的刷新函数
        #     def update_clustering_ui(text):
        #         # 逻辑 1：只有是 Single Cell Clustering 时才显示 Context Scale
        #         if row_context_scale:
        #             row_context_scale.setVisible(text in ["Single Cell Clustering", "Region Clustering"])
                
        #         # 逻辑 2：只要不是 None，就显示 Clusters (k)
        #         if row_k:
        #             row_k.setVisible(text != "None")
                    
        #         # >>> [新增逻辑 3]：如果切走的不是单细胞聚类，立刻释放模型！ >>>
        #         if text == "Single Cell Clustering":
        #             clear_cluster_models_cache(keep="lemon")  # 清理 Kaiko，保留 Lemon
        #         elif text == "Region Clustering":
        #             clear_cluster_models_cache(keep="kaiko")  # 清理 Lemon，保留 Kaiko
        #         else:
        #             clear_cluster_models_cache(keep=None)     # 选 Handcrafted 或 None 时，全清！
        #         # <<< [新增结束] <<<

        #     # 1. 绑定信号：当下拉菜单改变时，触发刷新
        #     cmb_clustering.currentTextChanged.connect(update_clustering_ui)
            
        #     # 2. 初始化状态：确保刚加载模型时显示正确
        #     update_clustering_ui(cmb_clustering.currentText())

        
# <<< [核心新增结束] <<<
    # ------------------------------------------Close---------------------------------------------------------
# ------------------------------------------Close---------------------------------------------------------
    def closeEvent(self, e):
        self._stop()
        
        # +++ 退出时杀掉可能在运行的实时倍数探测线程 +++
        if hasattr(self, 'mag_widget'):
            self.mag_widget.close_threads()
        if hasattr(self, 'mag_widget_compact'):
            self.mag_widget_compact.close_threads()
        # +++++++++++++++++++++++++++++++++++++++
# 🚀【关键新增】：因为放大窗口现在独立了，主程序关闭时需要手动把它也关掉
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