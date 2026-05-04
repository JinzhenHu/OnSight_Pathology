# custom_widgets/InferencePreviewWidget.py
import cv2
from PyQt6.QtWidgets import QWidget, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap


class InferencePreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.main_label = QLabel(self)
        self.main_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_label.setStyleSheet("background-color: #1f2a35; border: 1px solid #4a627a;")

        self.overview_label = QLabel(self)
        self.overview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overview_label.setStyleSheet("""
            QLabel {
                background-color: rgba(20, 20, 20, 220);
                border: 2px solid #ecf0f1;
                border-radius: 6px;
            }
        """)
        self.overview_label.hide()

        self._main_rgb = None
        self._overview_rgb = None
        self._compact_mode = False

        self.setMinimumSize(260, 260)

    def resizeEvent(self, event):
        self.main_label.setGeometry(0, 0, self.width(), self.height())

        pad = 10
        ow = min(180, max(120, self.width() // 3))
        oh = ow
        self.overview_label.setGeometry(self.width() - ow - pad, pad, ow, oh)

        self._refresh_main()
        self._refresh_overview()
        super().resizeEvent(event)

    def _to_pixmap(self, rgb_img):
        h, w, c = rgb_img.shape
        qimg = QImage(rgb_img.data, w, h, c * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def set_main_image(self, rgb_img, compact_mode=False):
        self._main_rgb = rgb_img.copy()
        self._compact_mode = compact_mode
        self._refresh_main()

    def set_overview_image(self, rgb_img):
        self._overview_rgb = rgb_img.copy()
        self.overview_label.show()
        self._refresh_overview()

    def clear_overview(self):
        self._overview_rgb = None
        self.overview_label.clear()
        self.overview_label.hide()

    def _refresh_main(self):
        if self._main_rgb is None:
            return
        pix = self._to_pixmap(self._main_rgb)
        scaled = pix.scaled(
            self.main_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.main_label.setPixmap(scaled)

    def _refresh_overview(self):
        if self._overview_rgb is None:
            return
        pix = self._to_pixmap(self._overview_rgb)
        scaled = pix.scaled(
            self.overview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.overview_label.setPixmap(scaled)