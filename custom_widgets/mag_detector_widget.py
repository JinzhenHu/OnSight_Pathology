import gc
import time
import torch
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QLabel, QMessageBox
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from utils import load_model
import settings

class MagDetectorThread(QThread):
    result_ready = pyqtSignal(float, str)
    error_occurred = pyqtSignal(str)

    def __init__(self, get_region_callback):
        super().__init__()
        self.get_region_callback = get_region_callback
        self.running = True

    def run(self):
        try:
            # 1. 获取模型元数据
            meta = settings.MODEL_METADATA.get("Magnification (VIT)")
            if not meta:
                self.error_occurred.emit("Magnification model metadata not found in settings.")
                return

            # 2. 加载模型（只加载一次，保持在显存中供实时循环使用）
            res = load_model(meta)
            model = res["model"]
            process_func = res["process_region_func"]
            using_gpu = res['using_gpu']

            # 3. 实时循环检测
            while self.running:
                # 动态获取当前选中的截图区域
                region = self.get_region_callback()
                if region is None:
                    time.sleep(0.5)
                    continue

                # 运行推理
                frame, res_txt, metrics = process_func(
                    region,
                    model=model,
                    metadata=meta,
                    additional_configs={}
                )

                continuous_mag = metrics.get("continuous_mag", 0.0)
                pred_cls = metrics.get("pred_cls", "Unknown")

                if self.running:
                    self.result_ready.emit(continuous_mag, pred_cls)
                    #self.result_ready.emit(pred_cls)

                # 限制检测频率为 2 FPS，防止挤爆 GPU
                time.sleep(0.2)

            # 4. 循环结束，清理显存释放给主模型
            if using_gpu:
                del model
                gc.collect()
                torch.cuda.empty_cache()

        except Exception as e:
            self.error_occurred.emit(str(e))

    def stop(self):
        self.running = False
        self.wait()

class MagDetectorWidget(QWidget):
    # 必须是 float 和 str 两个参数！
    mag_detected = pyqtSignal(float, str) 
    tracking_state_changed = pyqtSignal(bool)
    # ...

    def __init__(self, get_region_callback, parent=None):
        super().__init__(parent)
        self.get_region_callback = get_region_callback
        self.thread = None
        self.is_tracking = False

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # 开关按钮 (Checkable)
        self.btn_toggle = QPushButton("Start Real-time Mag")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.clicked.connect(self._on_toggle)

        # 结果标签
        self.lbl_result = QLabel("Not detected")
        self.lbl_result.setStyleSheet("color: grey;")
        # 固定最小宽度防止数字跳动时界面抖动
        self.lbl_result.setMinimumWidth(100) 

        self.layout.addWidget(self.btn_toggle)
        self.layout.addWidget(self.lbl_result)

    def _on_toggle(self, checked):
            if checked:
                region = self.get_region_callback()
                if not region:
                    QMessageBox.warning(self, "Warning", "Please select a screen region first!")
                    self.btn_toggle.setChecked(False)
                    return

                self.is_tracking = True
                # >>> [新增代码 2] 发送开启信号 >>>
                self.tracking_state_changed.emit(True)
                # <<< [新增结束] <<<
                
                self.btn_toggle.setText("Stop Real-time Mag ⏹️")
                self.btn_toggle.setStyleSheet("background-color: #e74c3c; color: white;") 
                self.lbl_result.setText("Loading model...")
                self.lbl_result.setStyleSheet("color: orange;")

                # 启动线程...
                self.thread = MagDetectorThread(self.get_region_callback)
                self.thread.result_ready.connect(self._on_result)
                self.thread.error_occurred.connect(self._on_error)
                self.thread.start()
            else:
                self.is_tracking = False
                # >>> [新增代码 3] 发送关闭信号 >>>
                self.tracking_state_changed.emit(False)
                # <<< [新增结束] <<<
                
                self.btn_toggle.setText("Start Real-time Mag 🔍")
                self.btn_toggle.setStyleSheet("") 
                self.lbl_result.setText("Stopped")
                self.lbl_result.setStyleSheet("color: grey;")
                
                if self.thread:
                    self.thread.stop()
                    self.thread = None

    def _on_result(self, continuous_mag, pred_cls):
        if self.is_tracking:
            self.lbl_result.setText(f"{continuous_mag:.2f}x ({pred_cls})")
            self.lbl_result.setStyleSheet("color: #1abc9c; font-weight: bold;") 

            # 必须把 pred_cls 也 emit 出去！
            self.mag_detected.emit(continuous_mag, pred_cls)

    def _on_error(self, err_msg):
        self.btn_toggle.setChecked(False)
        self._on_toggle(False)
        self.lbl_result.setText("Error")
        self.lbl_result.setStyleSheet("color: red;")
        QMessageBox.critical(self, "Error", f"Mag Detection failed:\n{err_msg}")
        
    def close_threads(self):
        """主窗口关闭时安全杀掉线程"""
        if self.thread and self.thread.isRunning():
            self.thread.stop()