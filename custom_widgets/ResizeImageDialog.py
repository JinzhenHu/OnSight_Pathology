from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, QDialog, QScrollArea
)
from PyQt6.QtGui import QPixmap, QImage, QStandardItem, QStandardItemModel, QAction
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRect, QSize
from pynput import mouse
import numpy as np
##############################################################################################################################################
# Enlarge Image Dialog (Live View + Metrics - 上下分屏版)
##############################################################################################################################################
class ResizableImageDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enlarged Inference View (Live)")
        self.setMinimumSize(700, 800) 
        
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10) 
        self.main_layout.setSpacing(10)

        # --- 1. 上半部分：图片显示区域 ---
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #2c3e50; border-radius: 4px;") 
        self.main_layout.addWidget(self.image_label, stretch=4) 
        
        # >>> [新增核心]：在 image_label 上面盖一个悬浮小图标签 >>>
        self.overlay_label = QLabel(self.image_label) # 以 image_label 为父节点，实现绝对定位悬浮
        self.overlay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overlay_label.setStyleSheet("""
            background-color: rgba(255, 255, 255, 240); 
            border: 1px solid #dcdde1; 
            border-radius: 8px;
        """)
        self.overlay_label.hide() # 初始隐藏
        self.latest_overlay_rgb = None
        # <<< [新增结束] <<<

        # --- 2. 下半部分：文字显示区域 ---
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

    # >>> [新增代码] 设置和动态更新 Overlay 的逻辑 >>>
    def set_overlay(self, result_rgb: np.ndarray):
        """接收来自主界面的阵列并更新小窗"""
        self.latest_overlay_rgb = result_rgb.copy()
        self.update_overlay()

    def hide_overlay(self):
        """当主界面点击 Stop 时，隐藏小窗"""
        self.latest_overlay_rgb = None
        self.overlay_label.hide()

    def update_overlay(self):
        """负责计算小图的位置并绘制"""
        if self.latest_overlay_rgb is not None:
            h, w, c = self.latest_overlay_rgb.shape
            # 确保内存连续，防止花屏
            rgb_data = np.ascontiguousarray(self.latest_overlay_rgb)
            qimg = QImage(rgb_data.data, w, h, c * w, QImage.Format.Format_RGB888)
            
            # 动态计算右上角小图的大小 (主图宽度的 1/3，但最小不低于100px)
            size = max(100, min(self.image_label.width(), self.image_label.height()) // 3)
            pixmap = QPixmap.fromImage(qimg).scaled(
                size, size, 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            
            self.overlay_label.setPixmap(pixmap)
            self.overlay_label.setFixedSize(size, size)
            # 固定在 image_label 的右上角，留出 10px 的呼吸边距
            self.overlay_label.move(self.image_label.width() - size - 10, 10)
            self.overlay_label.show()
        else:
            self.overlay_label.hide()
    # <<< [新增结束] <<<

    def resizeEvent(self, event):
        self.update_image()
        self.update_overlay() # 窗口大小改变时，重新计算小图的位置使其紧贴右上角
        super().resizeEvent(event)

    def update_image(self):
        if not self.original_pixmap.isNull():
            scaled_pixmap = self.original_pixmap.scaled(
                self.image_label.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
# ##############################################################################################################################################
# # Enlarge Image Dialog (Live View + Metrics - 上下分屏版)
# ##############################################################################################################################################
# class ResizableImageDialog(QDialog):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setWindowTitle("Enlarged Inference View (Live)")
#         # 既然是上下分屏，我们可以让初始高度高一点，宽度稍微窄一点
#         self.setMinimumSize(700, 800) 
        
#         # >>> [修改开始] 改回垂直布局 QVBoxLayout，实现上下分屏 >>>
#         self.main_layout = QVBoxLayout(self)
#         self.main_layout.setContentsMargins(10, 10, 10, 10) 
#         self.main_layout.setSpacing(10) # 上下两块区域中间的间距

#         # --- 1. 上半部分：图片显示区域 ---
#         self.image_label = QLabel()
#         self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         self.image_label.setStyleSheet("background-color: #2c3e50; border-radius: 4px;") 
#         # 图片区域占比较大，给它更多的 stretch (比如 3 或 4)
#         self.main_layout.addWidget(self.image_label, stretch=4) 
        
#         # --- 2. 下半部分：文字显示区域 ---
#         self.text_scroll = QScrollArea()
#         self.text_scroll.setWidgetResizable(True)
#         # 上下分屏时，我们需要限制的是“最小高度”，保证文字框不会被挤得看不见
#         self.text_scroll.setMinimumHeight(150) 
#         self.text_scroll.setStyleSheet("border: none; background-color: transparent;")
        
#         self.text_label = QLabel("Waiting for data...")
#         self.text_label.setTextFormat(Qt.TextFormat.RichText) 
#         self.text_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
#         self.text_label.setStyleSheet("background-color: white; color: black; padding: 10px; border-radius: 4px;")
#         self.text_label.setWordWrap(True)
        
#         self.text_scroll.setWidget(self.text_label)
#         self.main_layout.addWidget(self.text_scroll, stretch=1) # 文字区域占 1 份高度
#         # <<< [修改结束] <<<
        
#         self.original_pixmap = QPixmap()

#     def set_image(self, image: np.ndarray):
#         h, w, c = image.shape
#         qimg = QImage(image.data, w, h, c * w, QImage.Format.Format_RGB888)
#         self.original_pixmap = QPixmap.fromImage(qimg)
#         self.update_image()

#     def set_text(self, text: str):
#         self.text_label.setText(text)

#     def resizeEvent(self, event):
#         self.update_image()
#         super().resizeEvent(event)

#     def update_image(self):
#         if not self.original_pixmap.isNull():
#             # 缩放基准依然是 self.image_label.size()，这样图片就不会和下面的文字打架
#             scaled_pixmap = self.original_pixmap.scaled(
#                 self.image_label.size(), 
#                 Qt.AspectRatioMode.KeepAspectRatio, 
#                 Qt.TransformationMode.SmoothTransformation
#             )
#             self.image_label.setPixmap(scaled_pixmap)

