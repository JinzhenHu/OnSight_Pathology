"""
First-time info dialog explaining how the magnification detector works.
Shown the first time a user starts the Real-time Mag detector — explains
the difference between the integer (standard objective) value and the
continuous estimate, so users know which one to trust.
"""
import os
import sys
import json
import math
import random
import logging

from PyQt6.QtCore import Qt, QTimer, QRect, QPointF
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QPainterPath, QPolygonF
)
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QWidget, QSizePolicy,
)


# ---------------------------------------------------------------------------
# Settings I/O — keep in sync with DpiWarningDialog._settings_path()
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
        logging.warning(f"Could not save mag detector info preference: {e}")


SUPPRESS_KEY = "suppress_mag_detector_info"


# ============================================================================
# AnimatedMagOutputDemo — visual explanation of the two-value display
# ----------------------------------------------------------------------------
# ============================================================================
class AnimatedMagOutputDemo(QWidget):
    DURATIONS = {
        "idle":          1000,
        "show_drift":    3500,
        "show_trusted":  2500,
        "show_both":     2500,
    }

    BOX_WIDTH = 260
    BOX_HEIGHT = 56

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(190)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._stage = "idle"
        self._stage_elapsed = 0
        self._continuous_val = 40.0

        self._tick = QTimer(self)
        self._tick.timeout.connect(self._on_tick)
        self._tick.start(60)

        self._stage_timer = QTimer(self)
        self._stage_timer.setSingleShot(True)
        self._stage_timer.timeout.connect(self._advance_stage)

        QTimer.singleShot(300, lambda: self._enter_stage("idle"))

    # ---- Stage machine ----------------------------------------------------
    def _enter_stage(self, stage: str):
        self._stage = stage
        self._stage_elapsed = 0
        if stage == "idle":
            self._continuous_val = 40.0
        self._stage_timer.start(self.DURATIONS[stage])

    def _advance_stage(self):
        order = ["idle", "show_drift", "show_trusted", "show_both"]
        i = order.index(self._stage)
        next_stage = order[(i + 1) % len(order)]
        self._enter_stage(next_stage)

    def _on_tick(self):
        self._stage_elapsed += 60
        if self._stage in ("show_drift", "show_trusted", "show_both"):
            t = self._stage_elapsed / 200.0
            jitter = (math.sin(t * 1.3) * 2.5
                      + math.sin(t * 2.7) * 1.5
                      + random.uniform(-0.4, 0.4))
            self._continuous_val = 40.0 + jitter
        else:
            self._continuous_val = 40.0
        self.update()

    # ---- Painting ---------------------------------------------------------
    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        p.fillRect(self.rect(), QColor("#f5f5f7"))

        cx = self.width() // 2
        box_x = cx - self.BOX_WIDTH // 2
        box_y = 24
        box_rect = QRect(box_x, box_y, self.BOX_WIDTH, self.BOX_HEIGHT)

        show_drift_active   = self._stage in ("show_drift", "show_both")
        show_trusted_active = self._stage in ("show_trusted", "show_both")

        # Output box
        path = QPainterPath()
        path.addRoundedRect(float(box_rect.x()), float(box_rect.y()),
                            float(box_rect.width()), float(box_rect.height()),
                            8.0, 8.0)
        p.fillPath(path, QColor("white"))
        p.setPen(QPen(QColor("#dadce0"), 1.2))
        p.drawPath(path)

        # Text content
        cont_text = f"{self._continuous_val:.2f}x"
        int_text  = "(40X Magnification)"

        font_main = QFont("SF Pro Display", 14, QFont.Weight.DemiBold)
        if not font_main.exactMatch():
            font_main = QFont("Segoe UI", 14, QFont.Weight.DemiBold)
        p.setFont(font_main)

        fm = p.fontMetrics()
        cont_w = fm.horizontalAdvance(cont_text)
        gap_w  = fm.horizontalAdvance(" ")
        int_w  = fm.horizontalAdvance(int_text)
        total_w = cont_w + gap_w + int_w

        text_y = box_y + (self.BOX_HEIGHT + fm.ascent() - fm.descent()) // 2
        text_x = cx - total_w // 2

        # Continuous value
        cont_color = QColor("#e67e22") if show_drift_active else QColor("#1abc9c")
        p.setPen(cont_color)
        p.drawText(text_x, text_y, cont_text)
        if show_drift_active:
            self._draw_underline(p, text_x - 2, text_y + 5,
                                 cont_w + 4, QColor("#e67e22"))

        # Integer / bracketed part
        int_color = QColor("#27ae60") if show_trusted_active else QColor("#202124")
        p.setPen(int_color)
        p.drawText(text_x + cont_w + gap_w, text_y, int_text)
        if show_trusted_active:
            self._draw_underline(p, text_x + cont_w + gap_w - 2,
                                 text_y + 5, int_w + 4,
                                 QColor("#27ae60"))

        # Annotations with arrows
        if show_drift_active:
            self._draw_annotation(
                p,
                anchor_x=text_x + cont_w // 2,
                anchor_y=text_y + 12,
                direction="down-left",
                color=QColor("#e67e22"),
                line1="Continuous estimate",
                line2="(can drift, use as soft guide)",
            )
        if show_trusted_active:
            self._draw_annotation(
                p,
                anchor_x=text_x + cont_w + gap_w + int_w // 2,
                anchor_y=text_y + 12,
                direction="down-right",
                color=QColor("#27ae60"),
                line1="✓ Trusted prediction",
                line2="(use this one)",
            )

        p.end()

    def _draw_underline(self, p, x, y, w, color):
        p.setPen(QPen(color, 2.0))
        p.drawLine(x, y, x + w, y)

    def _draw_annotation(self, p, anchor_x, anchor_y, direction, color,
                         line1, line2):
        if direction == "down-left":
            label_x = anchor_x - 110
            label_y = anchor_y + 70
        else:
            label_x = anchor_x + 20
            label_y = anchor_y + 70

        # Curved arrow
        path = QPainterPath()
        path.moveTo(anchor_x, anchor_y + 4)
        ctrl_x = anchor_x + (-30 if direction == "down-left" else 30)
        ctrl_y = anchor_y + 30
        end_x = label_x + 50 if direction == "down-left" else label_x + 4
        end_y = label_y - 18
        path.quadTo(ctrl_x, ctrl_y, end_x, end_y)

        p.setPen(QPen(color, 1.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

        # Arrowhead
        head_size = 6
        dx = end_x - ctrl_x
        dy = end_y - ctrl_y
        length = max(1e-3, math.hypot(dx, dy))
        ux, uy = dx / length, dy / length
        px, py = -uy, ux
        tip   = QPointF(end_x, end_y)
        left  = QPointF(end_x - ux * head_size + px * head_size * 0.6,
                        end_y - uy * head_size + py * head_size * 0.6)
        right = QPointF(end_x - ux * head_size - px * head_size * 0.6,
                        end_y - uy * head_size - py * head_size * 0.6)
        p.setBrush(QBrush(color))
        p.drawPolygon(QPolygonF([tip, left, right]))

        # Label
        font_label = QFont("SF Pro Text", 9, QFont.Weight.DemiBold)
        if not font_label.exactMatch():
            font_label = QFont("Segoe UI", 9, QFont.Weight.DemiBold)
        p.setFont(font_label)
        p.setPen(color)
        p.drawText(label_x, label_y, line1)

        font_sub = QFont("SF Pro Text", 8)
        if not font_sub.exactMatch():
            font_sub = QFont("Segoe UI", 8)
        p.setFont(font_sub)
        p.setPen(QColor("#7d8a99"))
        p.drawText(label_x, label_y + 13, line2)


# ============================================================================
# Dialog
# ============================================================================
class MagDetectorInfoDialog(QDialog):
    """One-time explanation of how the magnification detector reports values."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About the Magnification Detector")
        self.setMinimumWidth(500)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 22, 26, 18)
        layout.setSpacing(14)

        # ---------- Header ----------
        import sys
        if sys.platform == "darwin":
            header_color = "#ffffff"   # white on macOS 
        else:
            header_color = "#2c3e50"   # navy on Windows (light background)

        header = QLabel("🔬 About the Magnification Detector")
        header.setStyleSheet(
            f"font-size: 14pt; font-weight: 600; color: {header_color};"
        )
        layout.addWidget(header)

        # ---------- Body (short — animation explains the rest) ----------
        body = QLabel(
            "<p style='line-height:1.5;'>"
            "OnSight detects four standard magnification levels: "
            "<b>5×, 10×, 20×, 40×</b>."
            "<br><br>"
            "When the detector is active, you'll see two values "
            "(see the example below):"
            "</p>"
        )
        body.setWordWrap(True)
        body.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(body)

        # ---------- Animated output demo ----------
        self.demo = AnimatedMagOutputDemo()
        layout.addWidget(self.demo)

        # ---------- Footer ----------
        footer = QHBoxLayout()
        self.chk_suppress = QCheckBox("Don't show this again")
        self.chk_suppress.setChecked(True)
        self.chk_suppress.setStyleSheet("color:#555;")
        footer.addWidget(self.chk_suppress)
        footer.addStretch()

        btn_done = QPushButton("Got it")
        btn_done.setDefault(True)
        btn_done.setStyleSheet(
            "QPushButton { padding:8px 22px; border-radius:5px;"
            "background:#27ae60; color:white; border:none; font-weight:600; }"
            "QPushButton:hover { background:#2ecc71; }"
        )
        btn_done.clicked.connect(self.accept)
        footer.addWidget(btn_done)
        layout.addLayout(footer)

    def is_suppress_checked(self) -> bool:
        return self.chk_suppress.isChecked()


# ---------------------------------------------------------------------------
# Public entry point — this is what mag_detector_widget imports
# ---------------------------------------------------------------------------
def maybe_show_mag_detector_info(parent=None) -> None:
    """Show the mag detector info dialog the first time a user starts tracking.

    User can tick 'Don't show again' to silence it (defaulted to checked).
    """
    settings = _load_settings()
    if settings.get(SUPPRESS_KEY, False):
        return

    dlg = MagDetectorInfoDialog(parent=parent)
    dlg.exec()

    if dlg.is_suppress_checked():
        settings[SUPPRESS_KEY] = True
        _save_settings(settings)