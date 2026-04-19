# Custom Enlarge Image Dialog for OnSight Pathology
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, QDialog, QScrollArea
)
from PyQt6.QtGui import QPixmap, QImage, QStandardItem, QStandardItemModel, QAction
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRect, QSize
from pynput import mouse
import numpy as np
##############################################################################################################################################
# Custom Enlarge Image Dialog for OnSight Pathology
##############################################################################################################################################
class ResizableImageDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enlarged Inference View (Live)")
        self.setMinimumSize(700, 800) 
        
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10) 
        self.main_layout.setSpacing(10)

        # --- 1. half top: image display area ---
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #2c3e50; border-radius: 4px;") 
        self.main_layout.addWidget(self.image_label, stretch=4) 
        

        self.overlay_label = QLabel(self.image_label) 
        self.overlay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overlay_label.setStyleSheet("""
            background-color: rgba(255, 255, 255, 240); 
            border: 1px solid #dcdde1; 
            border-radius: 8px;
        """)
        self.overlay_label.hide()
        self.latest_overlay_rgb = None


        # --- 2. half bottom: text display area ---
        self.text_scroll = QScrollArea()
        self.text_scroll.setWidgetResizable(True)
        self.text_scroll.setMinimumHeight(150) 
        self.text_scroll.setStyleSheet("border: none; background-color: transparent;")
        
        self.text_label = QLabel("Waiting for data...")
        self.text_label.setTextFormat(Qt.TextFormat.RichText) 
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.text_label.setStyleSheet("background-color: white; color: black; padding: 10px; border-radius: 4px;")
        self.text_label.setWordWrap(True)
        
        self.text_scroll.setWidget(self.text_label)
        self.main_layout.addWidget(self.text_scroll, stretch=1) 
        
        self.original_pixmap = QPixmap()

    def set_image(self, image: np.ndarray):
        h, w, c = image.shape
        qimg = QImage(image.data, w, h, c * w, QImage.Format.Format_RGB888)
        self.original_pixmap = QPixmap.fromImage(qimg)
        self.update_image()

    def set_text(self, text: str):
        self.text_label.setText(text)

    def set_overlay(self, result_rgb: np.ndarray):
        """receive the overlay result from the worker thread, store it, and trigger an update to redraw the overlay on top of the main image."""
        self.latest_overlay_rgb = result_rgb.copy()
        self.update_overlay()

    def hide_overlay(self):
        """when the main interface clicks Stop, hide the small window"""
        self.latest_overlay_rgb = None
        self.overlay_label.hide()

    def update_overlay(self):
        """calculate the position of the small overlay and draw it"""
        if self.latest_overlay_rgb is not None:
            h, w, c = self.latest_overlay_rgb.shape
            # Ensure memory is contiguous 
            rgb_data = np.ascontiguousarray(self.latest_overlay_rgb)
            qimg = QImage(rgb_data.data, w, h, c * w, QImage.Format.Format_RGB888)
            
            # Scale the overlay to be about 1/3 the size of the main image
            size = max(100, min(self.image_label.width(), self.image_label.height()) // 3)
            pixmap = QPixmap.fromImage(qimg).scaled(
                size, size, 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            
            self.overlay_label.setPixmap(pixmap)
            self.overlay_label.setFixedSize(size, size)

            self.overlay_label.move(self.image_label.width() - size - 10, 10)
            self.overlay_label.show()
        else:
            self.overlay_label.hide()

    def resizeEvent(self, event):
        self.update_image()
        self.update_overlay()
        super().resizeEvent(event)

    def update_image(self):
        if not self.original_pixmap.isNull():
            scaled_pixmap = self.original_pixmap.scaled(
                self.image_label.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)


