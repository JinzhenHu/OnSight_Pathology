from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QPropertyAnimation, pyqtProperty
from PyQt6.QtGui import QColor, QPainter, QBrush

class PulsingDot(QWidget):
    def __init__(self, color="orange", parent=None):
        super().__init__(parent)
        self._radius = 6
        self._color = QColor(color)

        # Pulse animation
        # self.anim = QPropertyAnimation(self, b"radius")
        # self.anim.setDuration(1000)
        # self.anim.setStartValue(6)
        # self.anim.setEndValue(10)
        # self.anim.setLoopCount(-1)
        # self.anim.start()

        self.setFixedSize(20, 20)  # Space for dot to grow/shrink

    def set_color(self, color):
        self._color = QColor(color)
        self.update()

    def get_radius(self):
        return self._radius

    def set_radius(self, r):
        self._radius = r
        self.update()

    radius = pyqtProperty(int, fget=get_radius, fset=set_radius)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QBrush(self._color))
        painter.setPen(Qt.PenStyle.NoPen)
        center = self.rect().center()
        painter.drawEllipse(center, self._radius, self._radius)
