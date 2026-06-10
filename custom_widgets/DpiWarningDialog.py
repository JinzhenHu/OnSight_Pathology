"""
First-run dialog that asks Windows users to set display scaling to 100%
before using the magnification detector. Skips itself on Mac/Linux and
on machines already at 100% scaling.
"""
import os
import sys
import json
import logging

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QFrame, QSpacerItem, QSizePolicy, QWidget,
)

# ---------------------------------------------------------------------------
# Settings I/O — kept self-contained so this dialog doesn't depend on the
# main app's settings object. Mirrors the path used by app.py.
# ---------------------------------------------------------------------------
def _settings_path() -> str:
    local_appdata = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    settings_dir = os.path.join(local_appdata, "OnSightPathology", "Settings")
    os.makedirs(settings_dir, exist_ok=True)
    return os.path.join(settings_dir, "settings.json")


def _load_settings() -> dict:
    try:
        with open(_settings_path(), "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_settings(data: dict) -> None:
    try:
        with open(_settings_path(), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        logging.warning(f"Could not save DPI warning preference: {e}")


# ---------------------------------------------------------------------------
# DPI detection
# ---------------------------------------------------------------------------
def _current_dpi_scale() -> float:
    """
    Return the system-wide display scaling as a ratio (1.0 = 100%, 1.25 = 125%).
    
    On Windows, query the OS directly via Win32 to avoid Qt's DPR which is
    affected by QT_SCALE_FACTOR and may snap to integer values.
    On Mac/Linux, fall back to Qt's reported ratio.
    """
    if sys.platform == "win32":
        try:
            import ctypes
            # GetDpiForSystem returns DPI as integer (96 = 100%, 120 = 125%, 144 = 150%, 192 = 200%)
            # Available on Windows 10 1607+
            user32 = ctypes.windll.user32
            try:
                dpi = user32.GetDpiForSystem()
                return dpi / 96.0
            except (AttributeError, OSError):
                # Older Windows fallback: use GetDeviceCaps on the screen DC
                gdi32 = ctypes.windll.gdi32
                LOGPIXELSX = 88
                hdc = user32.GetDC(0)
                dpi = gdi32.GetDeviceCaps(hdc, LOGPIXELSX)
                user32.ReleaseDC(0, hdc)
                return dpi / 96.0 if dpi else 1.0
        except Exception as e:
            logging.warning(f"Could not query Windows DPI: {e}")
            return 1.0
    else:
        return 1.0


# ---------------------------------------------------------------------------
# The dialog itself
# ---------------------------------------------------------------------------
class DpiWarningDialog(QDialog):
    """
    Friendly walkthrough explaining why and how to set Windows display
    scaling to 100% for accurate magnification detection.
    """

    SUPPRESS_KEY = "suppress_dpi_warning_mag"

    def __init__(self, current_scale: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Display Scaling Recommendation")
        self.setMinimumWidth(520)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 18)
        layout.setSpacing(14)

        # ---------- Header ----------
        header = QLabel("🖥️  Display Scaling Recommendation")
        header.setStyleSheet(
            "font-size: 15pt; font-weight: 600; color: #2c3e50;"
        )
        layout.addWidget(header)

        current_pct = int(round(current_scale * 100))
        intro = QLabel(
            f"<p style='line-height:1.5;'>"
            f"For accurate magnification detection, OnSight works best when "
            f"Windows display scaling is set to <b>100%</b>.<br>"
            f"Your display is currently at <b style='color:#e67e22;'>{current_pct}%</b>.</p>"
        )
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(intro)

        # ---------- Steps card ----------
        card = QFrame()
        card.setFrameShape(QFrame.Shape.StyledPanel)
        card.setStyleSheet(
            "QFrame { background:#f7f9fb; border:1px solid #d6dde4;"
            "border-radius:8px; padding:8px; }"
        )
        card_l = QVBoxLayout(card)
        card_l.setSpacing(12)
        card_l.setContentsMargins(14, 14, 14, 14)

        card_l.addWidget(self._step(
            "1",
            "Click the button below to open Windows Display Settings.",
        ))
        card_l.addWidget(self._step(
            "2",
            "Scroll to <b>Scale and layout</b>.",
            mock_html=(
                "<div style='font-family:Segoe UI, sans-serif; padding:6px 10px;"
                "background:white; border:1px solid #c8d2da; border-radius:4px;"
                "color:#2c3e50;'>"
                "<div style='font-size:8pt; color:#7d8a99;'>Scale</div>"
                "<div style='font-size:10pt;'>"
                f"<span style='color:#e67e22; font-weight:bold;'>{current_pct}%</span> "
                "<span style='color:#95a5a6;'>(Recommended) ▾</span>"
                "</div></div>"
            ),
        ))
        card_l.addWidget(self._step(
            "3",
            "Change it to <b>100%</b>.",
            mock_html=(
                "<div style='font-family:Segoe UI, sans-serif; padding:6px 10px;"
                "background:white; border:1px solid #27ae60; border-radius:4px;"
                "color:#2c3e50;'>"
                "<div style='font-size:8pt; color:#7d8a99;'>Scale</div>"
                "<div style='font-size:10pt;'>"
                "<span style='color:#27ae60; font-weight:bold;'>100%</span> "
                "<span style='color:#27ae60;'>✓</span>"
                "</div></div>"
            ),
        ))
        card_l.addWidget(self._step(
            "4",
            "<b>Restart OnSight</b> for the change to take full effect.",
        ))

        layout.addWidget(card)

        # ---------- Open Settings button ----------
        self.btn_open = QPushButton("  Open Display Settings  ↗")
        self.btn_open.setStyleSheet(
            "QPushButton { background:#3498db; color:white; padding:10px 16px;"
            "border:none; border-radius:6px; font-weight:600; font-size:11pt; }"
            "QPushButton:hover { background:#4ea8e1; }"
            "QPushButton:pressed { background:#2980b9; }"
        )
        self.btn_open.clicked.connect(self._open_display_settings)
        layout.addWidget(self.btn_open)

        # ---------- Footer: checkbox + Got it ----------
        footer = QHBoxLayout()
        self.chk_suppress = QCheckBox("Don't show this again")
        self.chk_suppress.setStyleSheet("color:#555;")
        footer.addWidget(self.chk_suppress)
        footer.addStretch()

        btn_ok = QPushButton("Got it")
        btn_ok.setDefault(True)
        btn_ok.setStyleSheet(
            "QPushButton { padding:8px 22px; border-radius:5px;"
            "background:#ecf0f1; color:#2c3e50; border:1px solid #bdc3c7; }"
            "QPushButton:hover { background:#dfe4e6; }"
        )
        btn_ok.clicked.connect(self.accept)
        footer.addWidget(btn_ok)
        layout.addLayout(footer)

    # ---------- Helpers ----------
    @staticmethod
    def _step(number: str, text_html: str, mock_html: str | None = None) -> QWidget:
        """A single step row: numbered badge on left, instruction + optional mock on right."""

        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(12)

        badge = QLabel(number)
        badge.setFixedSize(28, 28)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(
            "QLabel { background:#3498db; color:white; border-radius:14px;"
            "font-weight:bold; font-size:11pt; }"
        )
        h.addWidget(badge, alignment=Qt.AlignmentFlag.AlignTop)

        right = QVBoxLayout()
        right.setSpacing(6)
        right.setContentsMargins(0, 2, 0, 0)

        lbl_text = QLabel(text_html)
        lbl_text.setWordWrap(True)
        lbl_text.setTextFormat(Qt.TextFormat.RichText)
        lbl_text.setStyleSheet("color:#2c3e50; font-size:10pt;")
        right.addWidget(lbl_text)

        if mock_html:
            mock = QLabel(mock_html)
            mock.setTextFormat(Qt.TextFormat.RichText)
            mock.setStyleSheet("margin-top:2px;")
            right.addWidget(mock)

        h.addLayout(right, stretch=1)
        return row

    def _open_display_settings(self):
        """Open Windows 10/11 display settings via the ms-settings: protocol."""
        try:
            if sys.platform == "win32":
                os.startfile("ms-settings:display")  # opens Settings → System → Display
            else:
                # Shouldn't happen — dialog only shown on Windows, but be safe.
                logging.info("Display Settings only available on Windows")
        except Exception as e:
            logging.warning(f"Could not open Display Settings: {e}")

    def is_suppress_checked(self) -> bool:
        return self.chk_suppress.isChecked()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def maybe_show_dpi_warning(parent=None) -> None:
    """
    Show the DPI warning dialog if:
      - We are on Windows
      - The user hasn't ticked "Don't show this again"
      - The current display scaling is NOT already 100%
    
    Otherwise, do nothing.
    """
    if sys.platform != "win32":
        return

    settings = _load_settings()
    if settings.get(DpiWarningDialog.SUPPRESS_KEY, False):
        return

    scale = _current_dpi_scale()
    if abs(scale - 1.0) < 0.05:
        # Already at 100% (or close enough) — no warning needed.
        return

    dlg = DpiWarningDialog(current_scale=scale, parent=parent)
    dlg.exec()

    if dlg.is_suppress_checked():
        settings[DpiWarningDialog.SUPPRESS_KEY] = True
        _save_settings(settings)