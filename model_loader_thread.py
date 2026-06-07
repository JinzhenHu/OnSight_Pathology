"""
Background model loader with progress reporting.
Decouples slow HF downloads from the GUI thread.
"""
import logging
import traceback
from PyQt6.QtCore import QThread, pyqtSignal

import settings
from utils import load_model


class ModelLoaderThread(QThread):
    """
    Loads a model in the background.
    Emits progress updates and a final 'finished' signal with the loaded resources.
    """
    # signals
    progress = pyqtSignal(str, int, float, float)        # (status_text, percent 0..100; -1 = indeterminate, current_bytes, total_bytes)
    finished_ok = pyqtSignal(dict)           # the dict returned by load_model()
    failed = pyqtSignal(str, str)            # (short_msg, full_traceback)

    def __init__(self, model_name: str, parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self._abort = False

    def request_abort(self):
        self._abort = True

    def run(self):
        try:
            self.progress.emit("Preparing…", -1, 0, 0)
            self._install_hf_progress_hook()

            if self._abort:
                return

            model_info = settings.MODEL_METADATA[self.model_name]
            self.progress.emit(f"Loading {self.model_name}…", -1, 0, 0)

            res = load_model(model_info)

            if self._abort:
                # User cancelled but load actually finished — free the resources
                try:
                    import torch, gc
                    if res.get("using_gpu") and res.get("model") is not None:
                        del res["model"]
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                return  # Don't emit finished_ok

            self.progress.emit("Ready", 100, 0, 0)
            self.finished_ok.emit(res)

        except Exception as e:
            if self._abort:
                # Aborted-related exception is expected, no need to surface
                return
            import traceback
            logging.critical(f"Model load failed: {self.model_name}", exc_info=True)
            short = f"{type(e).__name__}: {str(e)[:300]}"
            self.failed.emit(short, traceback.format_exc())

    # ---------- HuggingFace progress hook ----------
    def _install_hf_progress_hook(self):
        try:
            outer = self
            
            # Reset session counters for this load
            _session_total = {"value": 0}
            _session_done = {"value": 0}

            class _ProgressTqdm:
                def __init__(self, *args, **kwargs):
                    self.total = kwargs.get("total") or 0
                    self.n = 0
                    self.desc = kwargs.get("desc", "Downloading")
                    self.disable = False
                    if self.total > 0:
                        _session_total["value"] += self.total

                def update(self, increment):
                    if outer._abort:
                        raise RuntimeError("User aborted download")
                    self.n += increment
                    if self.total > 0:
                        _session_done["value"] += increment
                        sess_total = _session_total["value"]
                        sess_done = _session_done["value"]
                        if sess_total > 0:
                            pct = int(sess_done / sess_total * 100)
                            outer.progress.emit(
                                "Downloading", pct, float(sess_done), float(sess_total)
                            )
                        else:
                            outer.progress.emit("Downloading…", -1, 0.0, 0.0)
                    else:
                        outer.progress.emit("Downloading…", -1, 0.0, 0.0)

                def close(self):
                    if self.total > 0 and self.n < self.total:
                        _session_done["value"] += (self.total - self.n)

                def set_description(self, desc, refresh=True):
                    self.desc = desc

                def reset(self, total=None):
                    if total:
                        _session_total["value"] += total
                    self.total = total or 0
                    self.n = 0

                def refresh(self):
                    pass

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    self.close()

                def __iter__(self):
                    return iter([])

                def __getattr__(self, name):
                    return lambda *a, **k: None

            # (Patch every sys.modules entry as before)
            import sys as _sys
            patched = []
            for mod_name, mod in list(_sys.modules.items()):
                if not mod_name.startswith("huggingface_hub"):
                    continue
                if mod is None:
                    continue
                for attr in ("tqdm", "hf_tqdm"):
                    if hasattr(mod, attr):
                        try:
                            setattr(mod, attr, _ProgressTqdm)
                            patched.append(f"{mod_name}.{attr}")
                        except Exception:
                            pass

            logging.info(f"HF progress hook installed at: {patched}")

            import os
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)

        except Exception as e:
            logging.warning(f"Could not install HF progress hook: {e}")