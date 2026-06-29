"""
Spinner dialog for fast model loads (bundled / cached).
"""
import math
import sys
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QWidget, QGraphicsDropShadowEffect,
    QPushButton
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSize, QRectF
from PyQt6.QtGui import QPainter, QPen, QColor, QConicalGradient, QPainterPath


SPINNER_STYLESHEET = """
QDialog {
    background-color: #ffffff;
    border-radius: 14px;
}

QLabel#SpinnerTitle {
    color: #1a1a1a;
    font-size: 13pt;
    font-weight: 600;
}

QLabel#SpinnerModel {
    color: #2c3e50;
    font-size: 10pt;
    font-weight: 500;
}

QLabel#SpinnerStatus {
    color: #6c757d;
    font-size: 9pt;
}

QLabel#SpinnerHint {
    color: #adb5bd;
    font-size: 8pt;
    font-style: italic;
}
"""


class _AnimatedSpinner(QWidget):
    """
    Custom-painted spinner — a rotating arc with a conical gradient
    that fades from the brand blue to transparent. Smoother and more
    'premium' than the QProgressBar busy indicator.
    """
    def __init__(self, parent=None, diameter=72):
        super().__init__(parent)
        self._diameter = diameter
        self._angle = 0
        self.setFixedSize(QSize(diameter, diameter))

        # 60 fps rotation
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)

    def _tick(self):
        self._angle = (self._angle + 6) % 360
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Outer track — light gray circle (the 'background' of the spinner)
        thickness = max(4, self._diameter // 12)
        rect = QRectF(thickness / 2, thickness / 2,
                      self._diameter - thickness,
                      self._diameter - thickness)

        track_pen = QPen(QColor("#eef2f7"), thickness)
        track_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(track_pen)
        p.drawArc(rect, 0, 360 * 16)

        # Rotating gradient arc — brand blue fading to transparent
        gradient = QConicalGradient(rect.center(), -self._angle)
        gradient.setColorAt(0.00, QColor(52, 152, 219, 255))   # #3498db full
        gradient.setColorAt(0.45, QColor(52, 152, 219, 80))
        gradient.setColorAt(0.85, QColor(52, 152, 219, 0))
        gradient.setColorAt(1.00, QColor(52, 152, 219, 0))

        arc_pen = QPen()
        arc_pen.setBrush(gradient)
        arc_pen.setWidth(thickness)
        arc_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(arc_pen)
        # Sweep ~ 300 degrees so the leading edge stays visible
        p.drawArc(rect, -self._angle * 16, 300 * 16)

        # Bright head dot at the leading edge of the arc — makes the
        # motion clearly visible at a glance.
        head_angle_rad = math.radians(-self._angle)
        cx = rect.center().x() + (rect.width() / 2) * math.cos(head_angle_rad)
        cy = rect.center().y() + (rect.height() / 2) * math.sin(head_angle_rad)

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#3498db"))
        dot_r = thickness / 2 + 1
        p.drawEllipse(QRectF(cx - dot_r, cy - dot_r,
                              dot_r * 2, dot_r * 2))

        p.end()

    def stop(self):
        self._timer.stop()


class SpinnerDialog(QDialog):
    """
    Lightweight 'loading' dialog used when the model is already cached
    locally — no progress bar, just an animation. On Windows/Linux it
    cannot be cancelled because the load is normally fast (1-3 seconds).
    On macOS a small close (×) button is shown in the top-right of the
    card so the user can dismiss the dialog if a large model takes
    longer than expected; the underlying loader keeps running and is
    cleaned up by the caller (see `cancelled` signal).
    """
    cancelled = pyqtSignal()   # fired only on macOS, when the × is clicked

    def __init__(self, title: str = "Loading Model",
                 model_name: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setFixedSize(360, 280)

        # Frameless rounded card look — matches premium dialogs in the app.
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.FramelessWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Outer wrapper so we can paint a rounded corner card with a shadow.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)

        card = QWidget(self)
        card.setObjectName("Card")
        card.setStyleSheet(SPINNER_STYLESHEET + """
            QWidget#Card {
                background-color: #ffffff;
                border-radius: 14px;
            }
        """)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(40)
        shadow.setOffset(0, 6)
        shadow.setColor(QColor(0, 0, 0, 60))
        card.setGraphicsEffect(shadow)

        outer.addWidget(card)

        # Inner layout
        v = QVBoxLayout(card)
        v.setContentsMargins(28, 28, 28, 24)
        v.setSpacing(14)
        v.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Spinner
        self.spinner = _AnimatedSpinner(card, diameter=72)
        v.addWidget(self.spinner, alignment=Qt.AlignmentFlag.AlignCenter)

        # Title
        self.lbl_title = QLabel(title, card)
        self.lbl_title.setObjectName("SpinnerTitle")
        self.lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self.lbl_title)

        # Model name
        self.lbl_model = QLabel(model_name, card)
        self.lbl_model.setObjectName("SpinnerModel")
        self.lbl_model.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_model.setWordWrap(True)
        v.addWidget(self.lbl_model)

        # Status (animated dots are appended by set_status)
        self.lbl_status = QLabel("Initializing", card)
        self.lbl_status.setObjectName("SpinnerStatus")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self.lbl_status)

        # Subtle hint
        self.lbl_hint = QLabel("This usually takes a few seconds", card)
        self.lbl_hint.setObjectName("SpinnerHint")
        self.lbl_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self.lbl_hint)

        # Animated "Initializing." → "Initializing.." → "Initializing..."
        self._base_status = "Initializing"
        self._dot_count = 0
        self._dot_timer = QTimer(self)
        self._dot_timer.timeout.connect(self._tick_dots)
        self._dot_timer.start(450)

        # macOS-only close (×) button. Frameless dialogs have no window
        # chrome, so on Mac we render a small inline × in the top-right
        # corner of the card. The button emits `cancelled` and dismisses
        # the dialog immediately; the caller is responsible for handling
        # the still-running loader thread (see app.py).
        self._card = card
        self.btn_close = None
        if sys.platform == "darwin":
            self.btn_close = QPushButton("×", card)
            self.btn_close.setObjectName("SpinnerClose")
            self.btn_close.setFixedSize(26, 26)
            self.btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
            self.btn_close.setToolTip("Close (the model will keep loading in the background)")
            self.btn_close.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.btn_close.setStyleSheet("""
                QPushButton#SpinnerClose {
                    background-color: transparent;
                    color: #adb5bd;
                    border: none;
                    border-radius: 13px;
                    font-size: 18pt;
                    font-weight: 300;
                    padding: 0px;
                    margin: 0px;
                }
                QPushButton#SpinnerClose:hover {
                    background-color: #f1f3f5;
                    color: #495057;
                }
                QPushButton#SpinnerClose:pressed {
                    background-color: #e9ecef;
                    color: #212529;
                }
                QPushButton#SpinnerClose:disabled {
                    color: #dee2e6;
                    background-color: transparent;
                }
            """)
            self.btn_close.clicked.connect(self._on_close_clicked)

    # ------------------------------------------------------------------
    # Public API kept compatible with LoadingDialog so callers can swap.
    # ------------------------------------------------------------------
    def set_model_name(self, name: str):
        self.lbl_model.setText(name)

    def set_status(self, text: str):
        self._base_status = text
        self._dot_count = 0
        self.lbl_status.setText(text)

    def set_progress(self, percent: int,
                     current_bytes: float = 0,
                     total_bytes: float = 0):
        # No-op: this dialog intentionally has no progress bar.
        pass

    def is_cancel_requested(self) -> bool:
        return False

    # ------------------------------------------------------------------
    def _tick_dots(self):
        self._dot_count = (self._dot_count + 1) % 4
        self.lbl_status.setText(self._base_status + "." * self._dot_count)

    # ------------------------------------------------------------------
    # macOS close button — placement + click handling
    # ------------------------------------------------------------------
    def showEvent(self, event):
        super().showEvent(event)
        # Position the × in the top-right corner of the card, with a
        # small inset. Done in showEvent so the card has its final size.
        if self.btn_close is not None and self._card is not None:
            card_w = self._card.width()
            inset = 8
            self.btn_close.move(card_w - self.btn_close.width() - inset, inset)
            self.btn_close.raise_()

    def _on_close_clicked(self):
        # Disable immediately so rapid double-clicks don't re-emit.
        if self.btn_close is not None:
            self.btn_close.setEnabled(False)
        # Stop our own animations so the dialog feels instantly responsive.
        try:
            self._dot_timer.stop()
        except Exception:
            pass
        try:
            self.spinner.stop()
        except Exception:
            pass
        self.cancelled.emit()
        # Close right away — the caller has hooked `cancelled` and will
        # take ownership of the still-running loader thread.
        self.reject()

    def closeEvent(self, event):
        self._dot_timer.stop()
        self.spinner.stop()
        super().closeEvent(event)