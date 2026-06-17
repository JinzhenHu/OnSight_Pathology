"""
LowMemoryWarningDialog.py
─────────────────────────
macOS-only warnings for low-RAM machines.

Two dialogs and two entry-point helpers:

1. LowMemoryWarningDialog — soft warning for PanNuke (CellViT).
   The user can dismiss it and continue at their own risk, or check
   "Don't show again for PanNuke" to silence future warnings.

2. UnsupportedOnMacDialog — hard block for the chat assistant
   (Lingshu-7B VLM). The dialog only offers an OK button — no
   "continue anyway", no "don't show again". The VLM is simply not
   supported on Macs with this little RAM.
"""

import sys
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox
)
from PyQt6.QtCore import Qt


# Trigger threshold. 8.5 gives a small slack so that 8 GB Macs (which
# psutil may report as ~7.8–8.0) consistently trigger, while 16 GB Macs
# (reported ~15.9) stay above. Tune here if needed in the future.
THRESHOLD_GB = 8.5

# Persistent settings key for the PanNuke "don't show again" toggle.
# The VLM dialog intentionally has no equivalent key.
SUPPRESS_PANNUKE_KEY = "suppress_low_memory_pannuke_warning"

# Soft brand palette — non-alarming, matches the rest of the app.
_WARNING_AMBER = "#B7791F"
_ERROR_RED     = "#A32D2D"
_BODY_TEXT     = "#444444"
_HINT_TEXT     = "#666666"


# ─────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────
def _detect_ram_gb() -> float:
    """
    Total system RAM in GB. Falls back to a large sentinel value on
    error so a probe failure can never accidentally lock users out of
    features that should otherwise be available.
    """
    try:
        from utils import get_system_memory
        return float(get_system_memory())
    except Exception:
        return 999.0


def is_low_memory_mac() -> bool:
    """True when running on macOS with detected RAM ≤ THRESHOLD_GB."""
    if sys.platform != "darwin":
        return False
    return _detect_ram_gb() <= THRESHOLD_GB


# ─────────────────────────────────────────────────────────────────────
# Dialog 1: soft warning for PanNuke (CellViT)
# ─────────────────────────────────────────────────────────────────────
class LowMemoryWarningDialog(QDialog):
    """
    Soft warning shown before PanNuke (CellViT) starts loading on a
    low-RAM Mac. Offers "Cancel" and "Continue anyway", plus an
    optional "Don't show again" checkbox.
    """

    def __init__(self, ram_gb: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Low memory warning")
        self.setModal(True)
        self.setMinimumWidth(470)

        self._proceed = False
        self._dont_show_again = False

        self._build_ui(ram_gb)

    def _build_ui(self, ram_gb: float):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(12)

        # Title row with a warning glyph
        title_row = QHBoxLayout()
        title_row.setSpacing(10)
        icon_lbl = QLabel("⚠")
        icon_lbl.setStyleSheet(
            f"color: {_WARNING_AMBER}; font-size: 22pt;"
        )
        title_lbl = QLabel("PanNuke may run slowly on this Mac")
        title_lbl.setStyleSheet("font-size: 13pt; font-weight: 500;")
        title_lbl.setWordWrap(True)
        title_row.addWidget(icon_lbl, alignment=Qt.AlignmentFlag.AlignTop)
        title_row.addWidget(title_lbl, stretch=1)
        layout.addLayout(title_row)

        # Body — factual and soft, no alarmist language.
        body = QLabel(
            f"This Mac has {ram_gb:.1f} GB of RAM. "
            "PanNuke (CellViT) is a large segmentation model and may run "
            "slowly or be briefly unresponsive on Macs with 8 GB or less. "
            "<br><br>"
            "For best performance we recommend a Mac with at least 16 GB "
            "of RAM, but you can continue if needed."
        )
        body.setTextFormat(Qt.TextFormat.RichText)
        body.setWordWrap(True)
        body.setStyleSheet(
            f"color: {_BODY_TEXT}; font-size: 10pt; line-height: 150%;"
        )
        layout.addWidget(body)

        # "Don't show again" toggle — only persists if the user actually
        # clicks Continue. Cancelling does NOT silence future warnings.
        self.chk_dont_show = QCheckBox("Don't show this again for PanNuke")
        self.chk_dont_show.setStyleSheet(
            f"font-size: 9pt; color: {_HINT_TEXT};"
        )
        layout.addWidget(self.chk_dont_show)

        # Button row
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setDefault(True)
        self.btn_cancel.clicked.connect(self._on_cancel)

        self.btn_continue = QPushButton("Continue anyway")
        self.btn_continue.clicked.connect(self._on_continue)

        for b in (self.btn_cancel, self.btn_continue):
            b.setStyleSheet("padding: 6px 16px;")

        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_continue)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------
    def _on_continue(self):
        self._proceed = True
        # Only honour "don't show again" when the user actually proceeds.
        # Silencing future warnings on a Cancel would be a bad surprise.
        self._dont_show_again = self.chk_dont_show.isChecked()
        self.accept()

    def _on_cancel(self):
        self._proceed = False
        self._dont_show_again = False
        self.reject()

    def is_proceed(self) -> bool:
        return self._proceed

    def is_dont_show_again(self) -> bool:
        return self._dont_show_again


# ─────────────────────────────────────────────────────────────────────
# Dialog 2: hard block for the chat assistant (Lingshu-7B VLM)
# ─────────────────────────────────────────────────────────────────────
class UnsupportedOnMacDialog(QDialog):
    """
    Hard-block dialog shown when the user tries to open the chat
    assistant on a low-RAM Mac. No override — the only action is
    to acknowledge and close.
    """

    def __init__(self, ram_gb: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Not supported on this Mac")
        self.setModal(True)
        self.setMinimumWidth(470)

        self._build_ui(ram_gb)

    def _build_ui(self, ram_gb: float):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(12)

        # Title row
        title_row = QHBoxLayout()
        title_row.setSpacing(10)
        icon_lbl = QLabel("✕")
        icon_lbl.setStyleSheet(
            f"color: {_ERROR_RED}; font-size: 22pt;"
        )
        title_lbl = QLabel("Chat assistant is not available on this Mac")
        title_lbl.setStyleSheet("font-size: 13pt; font-weight: 500;")
        title_lbl.setWordWrap(True)
        title_row.addWidget(icon_lbl, alignment=Qt.AlignmentFlag.AlignTop)
        title_row.addWidget(title_lbl, stretch=1)
        layout.addLayout(title_row)

        # Body — factual, reassures user the rest of the app still works.
        body = QLabel(
            f"This Mac has {ram_gb:.1f} GB of RAM. The chat assistant uses "
            "a large language model (Lingshu-7B) that requires at least "
            "16 GB of RAM to run reliably, so it is not available on this "
            "Mac."
            "<br><br>"
            "The other analysis features — cell segmentation, magnification "
            "detection, mitosis counting, Ki-67 — are unaffected and remain "
            "available."
        )
        body.setTextFormat(Qt.TextFormat.RichText)
        body.setWordWrap(True)
        body.setStyleSheet(
            f"color: {_BODY_TEXT}; font-size: 10pt; line-height: 150%;"
        )
        layout.addWidget(body)

        # OK button only — no checkbox, no override.
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_ok = QPushButton("OK")
        self.btn_ok.setDefault(True)
        self.btn_ok.setStyleSheet("padding: 6px 22px;")
        self.btn_ok.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_ok)
        layout.addLayout(btn_row)


# ─────────────────────────────────────────────────────────────────────
# Public entry points — call these from app.py
# ─────────────────────────────────────────────────────────────────────
def maybe_warn_pannuke(parent, settings: dict, save_callback) -> bool:
    """
    On a low-memory Mac, show the soft PanNuke warning before model
    loading begins.

    Parameters
    ----------
    parent : QWidget
        Parent for the modal dialog.
    settings : dict
        The app's persistent settings dict. Mutated in place when the
        user checks "don't show again".
    save_callback : callable
        Called with no arguments to persist `settings` after the
        "don't show again" toggle is set. Pass the app's
        `self._save_settings` here.

    Returns
    -------
    bool
        True  — the caller should continue loading the model.
                (Returned on non-Mac, on Macs with enough RAM, after a
                previous "don't show again", and after the user clicks
                "Continue anyway".)
        False — the user cancelled at the warning; the caller must NOT
                start loading.
    """
    if not is_low_memory_mac():
        return True
    if settings.get(SUPPRESS_PANNUKE_KEY, False):
        return True

    dlg = LowMemoryWarningDialog(ram_gb=_detect_ram_gb(), parent=parent)
    dlg.exec()

    if dlg.is_dont_show_again():
        settings[SUPPRESS_PANNUKE_KEY] = True
        try:
            save_callback()
        except Exception:
            # Persistence failure shouldn't crash the load flow.
            pass

    return dlg.is_proceed()


def block_vlm_if_unsupported(parent) -> bool:
    """
    On a low-memory Mac, show the hard-block dialog and return True
    so the caller can early-return from the chat handler.

    Returns
    -------
    bool
        True  — the chat assistant must NOT be opened on this Mac.
        False — proceed with the normal chat flow.
    """
    if not is_low_memory_mac():
        return False

    dlg = UnsupportedOnMacDialog(ram_gb=_detect_ram_gb(), parent=parent)
    dlg.exec()
    return True
