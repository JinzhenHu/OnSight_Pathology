"""
InferenceLoadingWidget.py — Premium "first inference loading" placeholder.

Displays an Apple HIG-styled animated indicator while a heavy model warms
up (CellViT, Retinanet, kaiko-ViT). Overlays its parent widget completely
and stays on top of other children.

Usage in app.py:

    from custom_widgets.InferenceLoadingWidget import InferenceLoadingWidget

    # Lazy-create when needed:
    self._inference_loader = InferenceLoadingWidget(self.lbl_img)
    self._inference_loader.start()

    # When first frame arrives:
    self._inference_loader.stop()
"""

import sys

from PyQt6.QtCore import (
    Qt, QPropertyAnimation, QEasingCurve, QEvent, pyqtProperty,
)
from PyQt6.QtGui import QFont, QColor, QPainter, QPen, QBrush
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QGraphicsDropShadowEffect,
)


# ============================================================================
# Apple HIG color palette
# ============================================================================
COLOR_PRIMARY        = "#0A84FF"
COLOR_TEXT           = "#1D1D1F"
COLOR_TEXT_SECONDARY = "#6E6E73"
COLOR_TEXT_TERTIARY  = "#86868B"
COLOR_BG_OVERLAY     = "#F4F6F8"   # matches lbl_img's default background
COLOR_BG_CARD        = "#FFFFFF"
COLOR_RING_TRACK     = "#E5E7EB"
COLOR_BORDER         = "#ECECEF"


# ============================================================================
# Spinner — Apple-style rotating arc
# ============================================================================
class _SpinnerRing(QWidget):
    """Continuously rotating circular arc."""

    def __init__(self, diameter=52, parent=None):
        super().__init__(parent)
        self.setFixedSize(diameter, diameter)
        self._diameter = diameter
        self._angle = 0

        self._anim = QPropertyAnimation(self, b"angle", self)
        self._anim.setDuration(1400)
        self._anim.setStartValue(0)
        self._anim.setEndValue(360)
        self._anim.setLoopCount(-1)
        self._anim.setEasingCurve(QEasingCurve.Type.Linear)

    def start(self):
        self._anim.start()

    def stop(self):
        self._anim.stop()

    @pyqtProperty(int)
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        self._angle = value
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        margin = 4
        size = self._diameter - 2 * margin
        stroke_w = 4

        # Background track — full circle, light gray
        pen_track = QPen(QColor(COLOR_RING_TRACK))
        pen_track.setWidth(stroke_w)
        pen_track.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_track)
        painter.drawEllipse(margin, margin, size, size)

        # Animated arc — Apple blue, rotating clockwise from 12 o'clock
        pen_arc = QPen(QColor(COLOR_PRIMARY))
        pen_arc.setWidth(stroke_w)
        pen_arc.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_arc)

        start_angle = (90 - self._angle) * 16
        span_angle = -110 * 16
        painter.drawArc(margin, margin, size, size, start_angle, span_angle)


# ============================================================================
# Animated dots — three pulsing dots under the headline
# ============================================================================
class _AnimatedDots(QWidget):
    """Three dots that fade in/out in sequence."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(44, 10)
        self._phase = 0.0

        self._anim = QPropertyAnimation(self, b"phase", self)
        self._anim.setDuration(1500)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(3.0)
        self._anim.setLoopCount(-1)
        self._anim.setEasingCurve(QEasingCurve.Type.Linear)

    def start(self):
        self._anim.start()

    def stop(self):
        self._anim.stop()

    @pyqtProperty(float)
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        self._phase = value
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)

        dot_size = 6
        spacing = 7
        total_w = 3 * dot_size + 2 * spacing
        start_x = (self.width() - total_w) // 2
        y = (self.height() - dot_size) // 2

        for i in range(3):
            distance = abs(self._phase - i)
            distance = min(distance, 3 - distance)
            opacity = max(50, int(255 - distance * 130))
            color = QColor(COLOR_PRIMARY)
            color.setAlpha(opacity)
            painter.setBrush(color)
            painter.drawEllipse(start_x + i * (dot_size + spacing), y,
                                dot_size, dot_size)


# ============================================================================
# Main overlay widget
# ============================================================================
class InferenceLoadingWidget(QWidget):
    """Polished overlay shown while the first inference is in progress.

    Sizes itself to cover its parent. Centered card with spinner + headline
    + subtitle + animated dots.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Light background that matches lbl_img's default style, so the
        # overlay reads as "this area is busy" rather than a floating panel
        self.setAutoFillBackground(False)
        self.setStyleSheet(
            f"InferenceLoadingWidget {{ background-color: {COLOR_BG_OVERLAY}; }}"
        )

        # Watch parent resize so we always cover it
        if parent is not None:
            parent.installEventFilter(self)

        self._build_ui()

        # Hide until start() is called
        self.hide()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        # Outer layout: vertical stretches center the card vertically
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.addStretch(1)

        # Horizontal wrap: stretches on both sides center the card horizontally
        h_wrap = QHBoxLayout()
        h_wrap.setContentsMargins(0, 0, 0, 0)
        h_wrap.addStretch(1)

        # The card itself
        card = QFrame()
        card.setObjectName("loadingCard")
        # Constrain dimensions so the card doesn't try to fill the whole area
        card.setFixedWidth(320)
        card.setMinimumHeight(220)
        card.setStyleSheet(
            f"#loadingCard {{ background-color: {COLOR_BG_CARD};"
            f"                border: 1px solid {COLOR_BORDER};"
            f"                border-radius: 16px; }}"
        )

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 35))
        card.setGraphicsEffect(shadow)

        h_wrap.addWidget(card)
        h_wrap.addStretch(1)
        outer.addLayout(h_wrap)
        outer.addStretch(1)

        # ---- Card contents ----
        content = QVBoxLayout(card)
        content.setContentsMargins(28, 26, 28, 22)
        content.setSpacing(0)

        # Spinner — centered
        spinner_row = QHBoxLayout()
        spinner_row.addStretch(1)
        self.spinner = _SpinnerRing(diameter=52)
        spinner_row.addWidget(self.spinner)
        spinner_row.addStretch(1)
        content.addLayout(spinner_row)
        content.addSpacing(16)

        # Headline
        headline = QLabel("Warming up the model")
        headline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        headline.setStyleSheet(
            f"font-size: 15px; font-weight: 600; color: {COLOR_TEXT};"
            f"background: transparent;"
        )
        content.addWidget(headline)
        content.addSpacing(6)

        # Subtitle
        subtitle = QLabel(
            "Heavy models take 5–30 seconds on first run.\n"
            "Subsequent frames will be much faster."
        )
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(
            f"font-size: 12px; color: {COLOR_TEXT_SECONDARY};"
            f"background: transparent;"
        )
        content.addWidget(subtitle)
        content.addSpacing(14)

        # Animated dots — centered
        dots_row = QHBoxLayout()
        dots_row.addStretch(1)
        self.dots = _AnimatedDots()
        dots_row.addWidget(self.dots)
        dots_row.addStretch(1)
        content.addLayout(dots_row)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self):
        """Begin animations, resize to parent, and show on top."""
        # CRITICAL: refresh geometry at show time. Parent may have been
        # tiny when __init__ ran, but is now properly laid out.
        if self.parent() is not None:
            self.setGeometry(self.parent().rect())

        self.spinner.start()
        self.dots.start()
        self.show()
        self.raise_()

    def stop(self):
        """Stop animations and hide the widget."""
        self.spinner.stop()
        self.dots.stop()
        self.hide()

    # ------------------------------------------------------------------
    # Track parent resize so we always cover it
    # ------------------------------------------------------------------
    def eventFilter(self, obj, event):
        if obj is self.parent() and event.type() == QEvent.Type.Resize:
            if self.isVisible():
                self.setGeometry(self.parent().rect())
        return super().eventFilter(obj, event)