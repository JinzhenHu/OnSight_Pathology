"""
mag_detector_widget.py — Real-time magnification detector with proper
model-loading UX.
"""

import gc
import time
import logging

import torch
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QLabel, QMessageBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer

from device_compat import empty_cache
from utils import load_model
import settings


# ============================================================================
# Phase 1: one-shot model loader (mirrors ModelLoaderThread's contract
# so we can reuse the existing LoadingDialog UI).
# ============================================================================
class MagLoaderThread(QThread):
    """Loads the magnification model once. Emits HF-style progress events.

    The model_loader_thread.ModelLoaderThread used by the main app is
    tightly coupled to the cmb_model dropdown; rather than refactor that,
    we provide a smaller compatible thread purpose-built for the mag model.
    """
    progress       = pyqtSignal(str, int, float, float)   # text, pct, cur_b, tot_b
    finished_ok    = pyqtSignal(dict)                     # {model, process_region_func, using_gpu}
    failed         = pyqtSignal(str, str)                 # short_msg, traceback
    load_mode_detected = pyqtSignal(str)                  # "download" | "cache"

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

            # We can't perfectly tell upfront whether the weights are
            # cached or need downloading, but load_model() is the slow
            # step regardless, so just announce "download" so the dialog
            # shows the full progress UI rather than a spinner.
            self.load_mode_detected.emit("download")
            self.progress.emit("Initializing magnification model…", -1, 0, 0)

            if self._abort:
                return

            # Actual load. This blocks; on a slow connection (or SSL
            # handshake retry) it can take 30-60s for the first download.
            res = load_model(meta)

            if self._abort:
                # User cancelled — drop the loaded model on the floor and
                # let GC reclaim it. Better than emitting finished_ok and
                # spinning up an inference loop the user no longer wants.
                try:
                    del res["model"]
                    gc.collect()
                    empty_cache()
                except Exception:
                    pass
                return

            self.progress.emit("Ready", 100, 0, 0)
            self.finished_ok.emit(res)

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logging.error("Mag detector model load failed", exc_info=True)

            # Translate the most common network errors into something
            # actionable for the end user. The full traceback is preserved
            # for the log file.
            err_str = str(e).lower()
            if "ssl" in err_str or "handshake" in err_str or "timed out" in err_str:
                short = ("Network timeout while downloading the magnification "
                         "model.\nCheck your internet connection and try again.")
            elif "connectionerror" in err_str or "getaddrinfo" in err_str:
                short = ("Could not reach huggingface.co.\n"
                         "If you're behind a firewall or in a region where HF "
                         "is throttled, try setting HF_ENDPOINT to a mirror.")
            elif "no space" in err_str or "disk" in err_str:
                short = "Not enough disk space to download the model."
            else:
                short = f"{type(e).__name__}: {str(e)[:200]}"

            self.failed.emit(short, tb)


# ============================================================================
# Phase 2: continuous inference loop. Model is already in GPU/CPU memory.
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
            model = self._res["model"]
            process_func = self._res["process_region_func"]
            using_gpu = self._res["using_gpu"]
            meta = settings.MODEL_METADATA["Magnification (VIT)"]

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
                    # Per-frame errors shouldn't kill the whole detector;
                    # log and try again next tick.
                    logging.warning(f"Mag detector frame failed: {e}")
                    time.sleep(0.5)
                    continue

                continuous_mag = metrics.get("continuous_mag", 0.0)
                pred_cls = metrics.get("pred_cls", "Unknown")

                if self.running:
                    self.result_ready.emit(continuous_mag, pred_cls)

                # ~2 FPS — fast enough to feel real-time, light enough to
                # leave room for the main inference thread.
                time.sleep(0.1)

            # Loop ended — release GPU memory
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
    mag_detected = pyqtSignal(float, str)
    tracking_state_changed = pyqtSignal(bool)

    def __init__(self, get_region_callback, parent=None):
        super().__init__(parent)
        self.get_region_callback = get_region_callback
        self.thread = None              # MagInferenceThread (after load)
        self._loader = None             # MagLoaderThread (during load)
        self._progress_dlg = None       # LoadingDialog
        self.is_tracking = False
        self._was_cancelled = False

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

    # ----------------------------------------------------------------------
    # Toggle: handles BOTH the initial click (start load) and the stop click.
    # ----------------------------------------------------------------------
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

        from custom_widgets.DpiWarningDialog import maybe_show_dpi_warning
        maybe_show_dpi_warning(parent=self.window(), context="mag")

        # --- Visual feedback ---
        self.btn_toggle.setText("Loading…")
        self.btn_toggle.setEnabled(False)
        self.lbl_result.setText("Loading model…")
        self.lbl_result.setStyleSheet("color: orange;")

        # --- Spawn the loading dialog (same widget as the main app uses) ---
        from custom_widgets.LoadingDialog import LoadingDialog

        self._was_cancelled = False
        self._progress_dlg = LoadingDialog(
            title="Loading Magnification Model",
            model_name="Magnification (VIT)",
            parent=self.window(),
        )

        self._loader = MagLoaderThread(parent=self)
        self._loader.progress.connect(self._on_load_progress)
        self._loader.finished_ok.connect(self._on_load_ok)
        self._loader.failed.connect(self._on_load_failed)
        self._loader.finished.connect(self._on_loader_finished)
        self._progress_dlg.cancelled.connect(self._on_cancel_load)

        self._loader.start()
        self._progress_dlg.show()

    def _stop_tracking(self):
        """Stop both the loader (if running) and the inference loop."""
        if self._loader is not None and self._loader.isRunning():
            # User clicked toggle off while loading — cancel the download
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

    # ----------------------------------------------------------------------
    # Loader event handlers
    # ----------------------------------------------------------------------
    def _on_load_progress(self, text, pct, cur_b, tot_b):
        if self._progress_dlg is None:
            return
        self._progress_dlg.set_status(text)
        self._progress_dlg.set_progress(pct, cur_b, tot_b)

    def _on_load_ok(self, model_resources):
        if self._was_cancelled:
            return
        self._close_progress_dialog()

        # --- Start inference now that the model is loaded ---
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

        # Reset UI
        self.btn_toggle.setChecked(False)
        self.btn_toggle.setText("Start Real-time Mag 🔍")
        self.btn_toggle.setStyleSheet("")
        self.btn_toggle.setEnabled(True)
        self.lbl_result.setText("Load failed")
        self.lbl_result.setStyleSheet("color: red;")

        # Use the same error dialog as the main app where available
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
        # Dialog stays open showing "Cancelling…" until _on_loader_finished.

    def _on_loader_finished(self):
        """Called after MagLoaderThread.run() returns, regardless of outcome."""
        if self._was_cancelled:
            self._close_progress_dialog()
            # Reset UI
            self.btn_toggle.setChecked(False)
            self.btn_toggle.setText("Start Real-time Mag 🔍")
            self.btn_toggle.setStyleSheet("")
            self.btn_toggle.setEnabled(True)
            self.lbl_result.setText("Cancelled")
            self.lbl_result.setStyleSheet("color: grey;")

        loader = self._loader
        self._loader = None
        if loader is not None:
            # Give Qt a tick to settle, then schedule C++ deletion via the
            # event loop instead of letting Python GC race with it.
            loader.wait(100)
            loader.deleteLater()

    def _close_progress_dialog(self):
        if self._progress_dlg is not None:
            try:
                self._progress_dlg.accept()
            except Exception:
                pass
            self._progress_dlg = None

    # ----------------------------------------------------------------------
    # Inference event handlers
    # ----------------------------------------------------------------------
    def _on_result(self, continuous_mag, pred_cls):
        if not self.is_tracking:
            return
        self.lbl_result.setText(f"{continuous_mag:.2f}x ({pred_cls})")
        self.lbl_result.setStyleSheet("color: #1abc9c; font-weight: bold;")
        self.mag_detected.emit(continuous_mag, pred_cls)

    def _on_inference_error(self, err_msg):
        # Inference loop crashed mid-run — show error and reset
        QMessageBox.critical(
            self.window(),
            "Mag Detection error",
            f"Detection failed during inference:\n{err_msg}",
        )
        self.btn_toggle.setChecked(False)
        self._stop_tracking()

    # ----------------------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------------------
    def close_threads(self):
        """Safely stop threads when the main window is closing."""
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