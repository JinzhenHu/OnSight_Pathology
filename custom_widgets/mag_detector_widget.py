#  Custom Widget for Real-time Magnification Detection in OnSight Pathology
import gc
import time
import torch
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QLabel, QMessageBox
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from device_compat import empty_cache
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
            # 1.get model metadata from settings 
            meta = settings.MODEL_METADATA.get("Magnification (VIT)")
            if not meta:
                self.error_occurred.emit("Magnification model metadata not found in settings.")
                return

            # 2. Load the model (this may take time, so we do it in the thread)
            res = load_model(meta)
            model = res["model"]
            process_func = res["process_region_func"]
            using_gpu = res['using_gpu']

            # 3. Real-time detection loop
            while self.running:
                # Dynamically get the currently selected screenshot region
                region = self.get_region_callback()
                if region is None:
                    time.sleep(0.5)
                    continue

                # Run inference
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

                # Limit detection frequency to 2 FPS to prevent GPU overload
                time.sleep(0.2)

            # 4. Loop ended, clean up GPU memory for the main model
            if using_gpu:
                del model
                gc.collect()
                empty_cache()

        except Exception as e:
            self.error_occurred.emit(str(e))

    def stop(self):
        self.running = False
        self.wait()

class MagDetectorWidget(QWidget):
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

        # Toggle button (Checkable)
        self.btn_toggle = QPushButton("Start Real-time Mag")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.clicked.connect(self._on_toggle)

        # Result label
        self.lbl_result = QLabel("Not detected")
        self.lbl_result.setStyleSheet("color: grey;")
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

                from custom_widgets.DpiWarningDialog import maybe_show_dpi_warning
                maybe_show_dpi_warning(parent=self.window())

                self.is_tracking = True
                self.tracking_state_changed.emit(True)

                self.btn_toggle.setText("Stop Real-time Mag ⏹️")
                self.btn_toggle.setStyleSheet("background-color: #e74c3c; color: white;") 
                self.lbl_result.setText("Loading model...")
                self.lbl_result.setStyleSheet("color: orange;")

                self.thread = MagDetectorThread(self.get_region_callback)
                self.thread.result_ready.connect(self._on_result)
                self.thread.error_occurred.connect(self._on_error)
                self.thread.start()
            else:
                self.is_tracking = False
                self.tracking_state_changed.emit(False)
                
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

            self.mag_detected.emit(continuous_mag, pred_cls)

    def _on_error(self, err_msg):
        self.btn_toggle.setChecked(False)
        self._on_toggle(False)
        self.lbl_result.setText("Error")
        self.lbl_result.setStyleSheet("color: red;")
        QMessageBox.critical(self, "Error", f"Mag Detection failed:\n{err_msg}")
        
    def close_threads(self):
        """Safely stop threads when the main window is closing."""
        if self.thread and self.thread.isRunning():
            self.thread.stop()