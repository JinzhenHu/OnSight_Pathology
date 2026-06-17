"""
LowMemoryWarningDialog.py
─────────────────────────
macOS-specific dialogs:

1. LowMemoryWarningDialog — soft warning for PanNuke (CellViT) on
   Macs with ≤ 8 GB RAM. User can choose "Continue anyway" or
   "Cancel", with an optional "Don't show again" toggle.

2. UnsupportedOnMacDialog — hard block for the chat assistant
   (Lingshu-7B VLM) on ALL Macs. Not RAM-gated — Lingshu-7B is not
   supported on macOS at all in the current build. No override,
   no "don't show again".


"""

import sys
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox,
    QFrame
)
from PyQt6.QtCore import Qt



PANNUKE_RAM_THRESHOLD_GB = 8.5

# Persistent settings key for the PanNuke "don't show again" toggle.
SUPPRESS_PANNUKE_KEY = "suppress_low_memory_pannuke_warning"

# ─────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────
_DIALOG_STYLESHEET = """
QDialog {
    background-color: #ffffff;
}
QDialog QWidget {
    background-color: #ffffff;
    color: #1a1a1a;
}
QDialog QLabel {
    background-color: transparent;
    color: #1a1a1a;
}
QDialog QLabel#TitleLabel {
    color: #1a1a1a;
    font-size: 14pt;
    font-weight: 600;
}
QDialog QLabel#BodyLabel {
    color: #2c2c2c;
    font-size: 11pt;
}
QDialog QLabel#IconLabel {
    background-color: transparent;
}
QDialog QCheckBox {
    color: #555555;
    font-size: 10pt;
    background-color: transparent;
    padding: 4px 0px;
}
QDialog QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #888888;
    border-radius: 3px;
    background-color: #ffffff;
}
QDialog QCheckBox::indicator:checked {
    background-color: #2563eb;
    border: 1px solid #2563eb;
}
QDialog QPushButton {
    background-color: #f0f0f0;
    color: #1a1a1a;
    border: 1px solid #c0c0c0;
    padding: 7px 18px;
    border-radius: 5px;
    font-size: 10pt;
    font-weight: 500;
    min-width: 90px;
}
QDialog QPushButton:hover {
    background-color: #e4e4e4;
    border: 1px solid #a0a0a0;
}
QDialog QPushButton:pressed {
    background-color: #d4d4d4;
}
QDialog QPushButton:default {
    background-color: #2563eb;
    color: #ffffff;
    border: 1px solid #1e4fc7;
}
QDialog QPushButton:default:hover {
    background-color: #1e54d6;
}
QDialog QPushButton:default:pressed {
    background-color: #1a47b8;
}
"""

# Accent colours used inline on the icon glyph only.
_WARNING_AMBER = "#B7791F"
_ERROR_RED     = "#A32D2D"


# ─────────────────────────────────────────────────────────────────────
# Detection helpers
# ─────────────────────────────────────────────────────────────────────
def _detect_ram_gb() -> float:
    """Total system RAM in GB. Returns a sentinel on failure so a
    probe error never accidentally locks users out of features that
    should otherwise be available."""
    try:
        from utils import get_system_memory
        return float(get_system_memory())
    except Exception:
        return 999.0


def is_mac() -> bool:
    """True on macOS."""
    return sys.platform == "darwin"


def is_low_memory_mac() -> bool:
    """True on macOS when detected RAM ≤ PANNUKE_RAM_THRESHOLD_GB."""
    if not is_mac():
        return False
    return _detect_ram_gb() <= PANNUKE_RAM_THRESHOLD_GB


# ─────────────────────────────────────────────────────────────────────
# Dialog 1: soft warning for PanNuke (CellViT)
# ─────────────────────────────────────────────────────────────────────
class LowMemoryWarningDialog(QDialog):
    """
    Soft warning shown before PanNuke (CellViT) starts loading on a
    low-RAM Mac. Lets the user proceed at their own risk or back out.
    """

    def __init__(self, ram_gb: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Low memory warning")
        self.setModal(True)
        self.setMinimumWidth(480)

        # Force our own stylesheet — do NOT inherit from the dark host.
        self.setStyleSheet(_DIALOG_STYLESHEET)
        self.setAutoFillBackground(True)

        self._proceed = False
        self._dont_show_again = False

        self._build_ui(ram_gb)

    def _build_ui(self, ram_gb: float):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 18)
        layout.setSpacing(14)

        # Title row with a warning glyph.
        title_row = QHBoxLayout()
        title_row.setSpacing(12)
        icon_lbl = QLabel("⚠")
        icon_lbl.setObjectName("IconLabel")
        icon_lbl.setStyleSheet(
            f"color: {_WARNING_AMBER}; font-size: 26pt; "
            f"background-color: transparent;"
        )
        icon_lbl.setFixedWidth(36)
        title_lbl = QLabel("PanNuke may run slowly on this Mac")
        title_lbl.setObjectName("TitleLabel")
        title_lbl.setWordWrap(True)
        title_row.addWidget(icon_lbl, alignment=Qt.AlignmentFlag.AlignTop)
        title_row.addWidget(title_lbl, stretch=1)
        layout.addLayout(title_row)

        # Body — factual, soft tone, no alarmist language.
        body = QLabel(
            f"This Mac has {ram_gb:.1f} GB of RAM. "
            "PanNuke (CellViT) is a large segmentation model and may run "
            "slowly or be briefly unresponsive on Macs with 8 GB or less."
            "<br><br>"
            "For best performance we recommend a Mac with at least 16 GB "
            "of RAM, but you can continue if needed."
        )
        body.setObjectName("BodyLabel")
        body.setTextFormat(Qt.TextFormat.RichText)
        body.setWordWrap(True)
        layout.addWidget(body)

        # Subtle divider above the don't-show-again toggle.
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("color: #e0e0e0; background-color: #e0e0e0;")
        divider.setFixedHeight(1)
        layout.addWidget(divider)

        # "Don't show again" toggle — only persisted on Continue.
        self.chk_dont_show = QCheckBox("Don't show this again for PanNuke")
        layout.addWidget(self.chk_dont_show)

        # Button row.
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self._on_cancel)

        self.btn_continue = QPushButton("Continue anyway")
        self.btn_continue.setDefault(True)
        self.btn_continue.clicked.connect(self._on_continue)

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
# Dialog 2: hard block for the chat assistant on macOS
# ─────────────────────────────────────────────────────────────────────
class UnsupportedOnMacDialog(QDialog):
    """
    Hard-block dialog shown when the user tries to open the chat
    assistant on macOS. No override — only an OK button.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Not supported on macOS")
        self.setModal(True)
        self.setMinimumWidth(480)

        # Force our own stylesheet — do NOT inherit from the dark host.
        self.setStyleSheet(_DIALOG_STYLESHEET)
        self.setAutoFillBackground(True)

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 18)
        layout.setSpacing(14)

        # Title row.
        title_row = QHBoxLayout()
        title_row.setSpacing(12)
        icon_lbl = QLabel("✕")
        icon_lbl.setObjectName("IconLabel")
        icon_lbl.setStyleSheet(
            f"color: {_ERROR_RED}; font-size: 26pt; "
            f"background-color: transparent;"
        )
        icon_lbl.setFixedWidth(36)
        title_lbl = QLabel("Chat assistant is not available on macOS")
        title_lbl.setObjectName("TitleLabel")
        title_lbl.setWordWrap(True)
        title_row.addWidget(icon_lbl, alignment=Qt.AlignmentFlag.AlignTop)
        title_row.addWidget(title_lbl, stretch=1)
        layout.addLayout(title_row)

        # Body — factual, reassures user the rest of the app works.
        body = QLabel(
            "The chat assistant relies on a CUDA-capable NVIDIA GPU for "
            "4-bit quantised inference of the Lingshu-7B model, which is "
            "not available on macOS in this build of OnSight."
            "<br><br>"
            "The other analysis features — cell segmentation, "
            "magnification detection, mitosis counting, Ki-67 — are "
            "unaffected and remain fully available on this Mac."
        )
        body.setObjectName("BodyLabel")
        body.setTextFormat(Qt.TextFormat.RichText)
        body.setWordWrap(True)
        layout.addWidget(body)

        # OK button only — no checkbox, no override.
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_ok = QPushButton("OK")
        self.btn_ok.setDefault(True)
        self.btn_ok.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_ok)
        layout.addLayout(btn_row)


# ─────────────────────────────────────────────────────────────────────
# Public entry points — called from app.py
# ─────────────────────────────────────────────────────────────────────
def maybe_warn_pannuke(parent, settings: dict, save_callback) -> bool:
    """
    On a low-memory Mac (≤ PANNUKE_RAM_THRESHOLD_GB), show the soft
    PanNuke warning before model loading begins.

    Parameters
    ----------
    parent : QWidget
    settings : dict
        The app's persistent settings dict. Mutated in place when the
        user checks "don't show again".
    save_callback : callable
        Called with no arguments to persist `settings` after the
        "don't show again" toggle is set. Pass `self._save_settings`.

    Returns
    -------
    bool
        True  — caller may proceed (non-Mac, enough RAM, previously
                dismissed, or user clicked "Continue anyway").
        False — user clicked Cancel; caller must NOT start loading.
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
    On ANY macOS machine (regardless of RAM), show the hard-block
    dialog and return True so the caller can early-return from the
    chat handler. The chat assistant is not supported on macOS.

    Returns
    -------
    bool
        True  — caller MUST NOT open the VLM.
        False — proceed normally (Windows / Linux).
    """
    if not is_mac():
        return False

    dlg = UnsupportedOnMacDialog(parent=parent)
    dlg.exec()
    return True