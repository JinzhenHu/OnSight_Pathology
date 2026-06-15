"""
mag_detector_widget.py — Real-time magnification detector with simple
indeterminate-spinner loading UX.
"""

import gc
import time
import logging

import torch
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QLabel, QMessageBox, QApplication,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer

from device_compat import empty_cache
from utils import load_model
import settings


# ============================================================================
# Phase 1: one-shot model loader.
# ============================================================================
class MagLoaderThread(QThread):
    finished_ok = pyqtSignal(dict)
    failed      = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._abort = False

    def request_abort(self):
        self._abort = True

    def run(self):
        try:
            meta = settings.MODEL_METADATA.get("Magnification (VIT)")
            if not meta:
                self.failed.emit(
                    "Magnification model metadata not found.",
                    "settings.MODEL_METADATA['Magnification (VIT)'] is missing"
                )
                return

            if self._abort:
                return

            res = load_model(meta)

            if self._abort:
                try:
                    del res["model"]
                    gc.collect()
                    empty_cache()
                except Exception:
                    pass
                return

            self.finished_ok.emit(res)

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logging.error("Mag detector model load failed", exc_info=True)

            err_str = str(e).lower()
            if "ssl" in err_str or "handshake" in err_str or "timed out" in err_str:
                short = ("Network timeout while downloading the magnification "
                         "model.\n\nCheck your internet connection and try again.")
            elif "connectionerror" in err_str or "getaddrinfo" in err_str:
                short = ("Could not reach huggingface.co.\n\n"
                         "If you're behind a firewall or in a region where HF "
                         "is throttled, try connecting to a different network.")
            elif "no space" in err_str or "disk" in err_str:
                short = "Not enough disk space to download the model."
            else:
                short = f"{type(e).__name__}: {str(e)[:200]}"

            self.failed.emit(short, tb)


# ============================================================================
# Phase 2: continuous inference loop.
# ============================================================================
class MagInferenceThread(QThread):
    result_ready   = pyqtSignal(float, str)
    error_occurred = pyqtSignal(str)

    def __init__(self, model_resources, get_region_callback, parent=None):
        super().__init__(parent)
        self._res = model_resources
        self.get_region_callback = get_region_callback
        self.running = True

    def run(self):
        try:
            model        = self._res["model"]
            process_func = self._res["process_region_func"]
            using_gpu    = self._res["using_gpu"]
            meta         = settings.MODEL_METADATA["Magnification (VIT)"]

            while self.running:
                region = self.get_region_callback()
                if region is None:
                    time.sleep(0.5)
                    continue

                try:
                    frame, res_txt, metrics = process_func(
                        region, model=model, metadata=meta, additional_configs={}
                    )
                except Exception as e:
                    logging.warning(f"Mag detector frame failed: {e}")
                    time.sleep(0.5)
                    continue

                continuous_mag = metrics.get("continuous_mag", 0.0)
                pred_cls       = metrics.get("pred_cls", "Unknown")

                if self.running:
                    self.result_ready.emit(continuous_mag, pred_cls)

                time.sleep(0.2)

            if using_gpu:
                try:
                    del model
                    gc.collect()
                    empty_cache()
                except Exception as e:
                    logging.warning(f"Mag GPU cleanup failed: {e}")

        except Exception as e:
            logging.error("Mag inference thread crashed", exc_info=True)
            self.error_occurred.emit(str(e))

    def stop(self):
        self.running = False


# ============================================================================
# Widget
# ============================================================================
class MagDetectorWidget(QWidget):
    mag_detected           = pyqtSignal(float, str)
    tracking_state_changed = pyqtSignal(bool)

    def __init__(self, get_region_callback, parent=None):
        super().__init__(parent)
        self.get_region_callback = get_region_callback
        self.thread          = None
        self._loader         = None
        self._progress_dlg   = None
        self.is_tracking     = False
        self._was_cancelled  = False

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.btn_toggle = QPushButton("Start Real-time Mag")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.clicked.connect(self._on_toggle)

        self.lbl_result = QLabel("Not detected")
        self.lbl_result.setStyleSheet("color: grey;")
        self.lbl_result.setMinimumWidth(100)

        self.layout.addWidget(self.btn_toggle)
        self.layout.addWidget(self.lbl_result)

    def _on_toggle(self, checked):
        if checked:
            self._start_loading()
        else:
            self._stop_tracking()

    def _start_loading(self):
        region = self.get_region_callback()
        if not region:
            QMessageBox.warning(self, "Warning", "Please select a screen region first!")
            self.btn_toggle.setChecked(False)
            return
        # ------------------------------------------------------------------
        # Show the one-time detector info dialog first — it explains the
        # two values (bracketed vs continuous) so users know which to trust.
        # Suppressed after the user dismisses it once (checkbox defaults to
        # "Don't show again" being ticked).
        # ------------------------------------------------------------------
        from custom_widgets.MagDetectorInfoDialog import maybe_show_mag_detector_info
        maybe_show_mag_detector_info(parent=self.window())


        # ------------------------------------------------------------------
        # Show DPI warning FIRST and check for restart-in-progress before
        # touching the spinner or loader. The DPI dialog's "Restart" button
        # calls subprocess.Popen() to spawn a new OnSight instance, then
        # QApplication.quit() — but quit() is ASYNC, so without this guard
        # the function continues executing, creating a SpinnerDialog and
        # starting a download. The new process then ALSO creates its own
        # SpinnerDialog, leaving two spinners on screen during the restart.
        #
        # The fix: detect the restart-in-progress case via closingDown()
        # and bail out before any further UI setup.
        # ------------------------------------------------------------------
        from custom_widgets.DpiWarningDialog import maybe_show_dpi_warning
        maybe_show_dpi_warning(parent=self.window(), context="mag")

        if self._is_app_quitting():
            # Old process is about to exit — fresh process will start clean.
            self.btn_toggle.setChecked(False)
            self.btn_toggle.setText("Start Real-time Mag")
            self.btn_toggle.setStyleSheet("")
            self.lbl_result.setText("Restarting…")
            self.lbl_result.setStyleSheet("color: grey;")
            return

        # --- Visual feedback ---
        self.btn_toggle.setText("Loading…")
        self.btn_toggle.setEnabled(False)
        self.lbl_result.setText("Loading model…")
        self.lbl_result.setStyleSheet("color: orange;")

        # --- Spinner dialog (indeterminate) ---
        from custom_widgets.SpinnerDialog import SpinnerDialog
        self._was_cancelled = False
        self._progress_dlg = SpinnerDialog(
            title="Loading Magnification Model",
            model_name="Magnification (VIT)",
            parent=self.window(),
        )
        self._progress_dlg.set_status(
            "Downloading from HuggingFace if not cached…\n"
            "This typically takes 5–30 seconds on first run."
        )
        self._progress_dlg.cancelled.connect(self._on_cancel_load)

        # --- Start the loader thread ---
        self._loader = MagLoaderThread(parent=self)
        self._loader.finished_ok.connect(self._on_load_ok)
        self._loader.failed.connect(self._on_load_failed)
        self._loader.finished.connect(self._on_loader_finished)

        self._loader.start()
        self._progress_dlg.show()

    @staticmethod
    def _is_app_quitting() -> bool:
        """Return True if QApplication.quit() has been called.

        Used to short-circuit further UI setup when the DPI warning has
        triggered a restart. QApplication.closingDown() flips to True once
        quit() initiates the shutdown sequence — even though quit() returns
        immediately, the flag is set synchronously.
        """
        app = QApplication.instance()
        if app is None:
            return True
        try:
            if app.closingDown():
                return True
        except Exception:
            pass
        return False

    def _stop_tracking(self):
        if self._loader is not None and self._loader.isRunning():
            self._on_cancel_load()
            return

        self.is_tracking = False
        self.tracking_state_changed.emit(False)

        self.btn_toggle.setText("Start Real-time Mag 🔍")
        self.btn_toggle.setStyleSheet("")
        self.btn_toggle.setEnabled(True)
        self.lbl_result.setText("Stopped")
        self.lbl_result.setStyleSheet("color: grey;")

        if self.thread is not None:
            self.thread.stop()
            self.thread.wait(2000)
            self.thread = None

    def _on_load_ok(self, model_resources):
        if self._was_cancelled:
            return
        self._close_progress_dialog()

        self.is_tracking = True
        self.tracking_state_changed.emit(True)

        self.btn_toggle.setText("Stop Real-time Mag ⏹️")
        self.btn_toggle.setStyleSheet("background-color: #e74c3c; color: white;")
        self.btn_toggle.setEnabled(True)

        self.thread = MagInferenceThread(
            model_resources, self.get_region_callback, parent=self
        )
        self.thread.result_ready.connect(self._on_result)
        self.thread.error_occurred.connect(self._on_inference_error)
        self.thread.start()

    def _on_load_failed(self, short_msg, traceback_str):
        self._close_progress_dialog()
        if self._was_cancelled:
            return

        self.btn_toggle.setChecked(False)
        self.btn_toggle.setText("Start Real-time Mag 🔍")
        self.btn_toggle.setStyleSheet("")
        self.btn_toggle.setEnabled(True)
        self.lbl_result.setText("Load failed")
        self.lbl_result.setStyleSheet("color: red;")

        try:
            from crash_logging import show_error_dialog
            show_error_dialog(
                title="Could not load magnification model",
                short_msg=short_msg,
                details=traceback_str,
                hint_html="",
                parent=self.window(),
            )
        except Exception:
            QMessageBox.critical(
                self.window(),
                "Mag Detection failed",
                short_msg,
            )

    def _on_cancel_load(self):
        self._was_cancelled = True
        if self._loader and self._loader.isRunning():
            self._loader.request_abort()

    def _on_loader_finished(self):
        if self._was_cancelled:
            self._close_progress_dialog()
            self.btn_toggle.setChecked(False)
            self.btn_toggle.setText("Start Real-time Mag 🔍")
            self.btn_toggle.setStyleSheet("")
            self.btn_toggle.setEnabled(True)
            self.lbl_result.setText("Cancelled")
            self.lbl_result.setStyleSheet("color: grey;")

        loader = self._loader
        self._loader = None
        if loader is not None:
            loader.wait(100)
            loader.deleteLater()

    def _close_progress_dialog(self):
        if self._progress_dlg is not None:
            try:
                self._progress_dlg.accept()
            except Exception:
                pass
            self._progress_dlg = None

    def _on_result(self, continuous_mag, pred_cls):
        if not self.is_tracking:
            return
        self.lbl_result.setText(f"{continuous_mag:.2f}x ({pred_cls})")
        self.lbl_result.setStyleSheet("color: #1abc9c; font-weight: bold;")
        self.mag_detected.emit(continuous_mag, pred_cls)

    def _on_inference_error(self, err_msg):
        QMessageBox.critical(
            self.window(),
            "Mag Detection error",
            f"Detection failed during inference:\n{err_msg}",
        )
        self.btn_toggle.setChecked(False)
        self._stop_tracking()

    def close_threads(self):
        if self._loader is not None and self._loader.isRunning():
            try:
                self._loader.request_abort()
                self._loader.wait(3000)
            except Exception:
                pass

        if self.thread is not None and self.thread.isRunning():
            try:
                self.thread.stop()
                self.thread.wait(3000)
            except Exception:
                pass