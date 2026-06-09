"""
Spinner dialog for fast model loads (bundled / cached).
No progress bar — just a clean, polished animation that says
'something is happening, please wait a moment'.
"""
import math
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QWidget, QGraphicsDropShadowEffect
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
    locally — no progress bar, just an animation. Cannot be cancelled
    because the load is fast (1-3 seconds).
    """
    cancelled = pyqtSignal()   # never fired, exists for API parity with LoadingDialog

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

    def closeEvent(self, event):
        self._dot_timer.stop()
        self.spinner.stop()
        super().closeEvent(event)
