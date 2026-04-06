import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame, QGraphicsDropShadowEffect
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QColor
import cv2
from utils import (get_tissue_mask, get_he_deconvolution, compute_suspicious_score_map, visualize_hotspot_overlay)

class OverlayWorker(QThread):
    """Background thread to process the overlay."""
    finished = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.frame = None
        self.is_busy = False

    # def run(self):
    #     self.is_busy = True
    #     try:
    #         tissue_mask = get_tissue_mask(self.frame)
    #         deconv, imH, imE = get_he_deconvolution(self.frame)
    #         score_map = compute_suspicious_score_map(self.frame, imH, imE, tissue_mask,sigma_region=1)
            
    #         # >>> [修改] 动态获取 percentile，默认给 80 >>>
    #         p_val = getattr(self, 'percentile', 80)
            
    #         result = visualize_hotspot_overlay(self.frame, score_map, tissue_mask, percentile=p_val)
    #         # <<< [修改结束] <<<
            
    #         self.finished.emit(result)
    #     except Exception as e:
    #         print(f"[Overlay Error]: {e}")
    #     finally:
    #         self.is_busy = False

    def run(self):
            self.is_busy = True
            try:
                # >>> [防呆修复 1] 无论外界传进来什么，强制校验并转为真正的 RGB >>>
                # 简单校验：判断图像的红色和蓝色通道均值，通常病理图偏红粉色(R>B)
                # 如果发现 B > R，说明极大可能传错了，是个 BGR 图，我们把它翻转过来！
                # mean_b = np.mean(self.frame[:, :, 2])
                # mean_r = np.mean(self.frame[:, :, 0])
                # if mean_b > mean_r:
                #     print("[OverlayWorker] Detected BGR format, converting to RGB.")
                #     self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                # # <<< [防呆修复结束] <<<
                k_size = getattr(self, 'kernel_size', 5)

                tissue_mask = get_tissue_mask(self.frame, kernel_size=k_size)
                deconv, imH, imE = get_he_deconvolution(self.frame)
                score_map = compute_suspicious_score_map(self.frame, imH, imE, tissue_mask, sigma_region=1)
                
                p_val = getattr(self, 'percentile', 80)
                result = visualize_hotspot_overlay(self.frame, score_map, tissue_mask, percentile=p_val)

                cv2.imwrite("debug_overlay.png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR)) # Debug: Save the overlay result to disk for inspection
                
                self.finished.emit(result)
            except Exception as e:
                print(f"[Overlay Error]: {e}")
            finally:
                self.is_busy = False
class OverlayWidget(QFrame):
    """The floating widget containing only the thumbnail image."""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Modern, clean styling for a white background environment
        self.setStyleSheet("""
            OverlayWidget {
                background-color: rgba(255, 255, 255, 240); /* Slightly transparent white */
                border: 1px solid #dcdde1; /* Very soft silver/gray border */
                border-radius: 8px; /* Smooth rounded corners */
            }
        """)
        
        # Add a subtle drop shadow to make it "float"
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 40)) # Soft 15% opacity black shadow
        shadow.setOffset(0, 4) # Cast the shadow slightly downwards
        self.setGraphicsEffect(shadow)
        
        self.state = "STOPPED"
        self.worker = OverlayWorker()
        self.worker.finished.connect(self._on_worker_finished)

        self.layout = QVBoxLayout(self)
        # 4px padding so the image breathes and doesn't clip the rounded borders
        self.layout.setContentsMargins(4, 4, 4, 4)
        
        self.lbl_thumbnail = QLabel()
        self.lbl_thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Transparent background for the image itself, with rounded inner corners
        self.lbl_thumbnail.setStyleSheet("""
            QLabel {
                background-color: transparent; 
                color: #2c3e50; 
                border: none;
                border-radius: 4px;
            }
        """)
        self.layout.addWidget(self.lbl_thumbnail)

    def update_size(self):
        """Dynamically scale to ~1/4 the size of the parent inference window."""
        if self.parent():
            pw, ph = self.parent().width(), self.parent().height()
            if pw < 50 or ph < 50:
                size = 80
            else:
                size = min(pw, ph) // 2 
                
            self.setFixedSize(size, size)

    def start(self):
        self.state = "RUNNING"
        self.update_size()
        self.show()

    def pause(self):
        self.state = "PAUSED"

    def stop(self):
        self.state = "STOPPED"
        self.hide()
        self.lbl_thumbnail.clear()

    def process_frame(self, frame_rgb):
        if self.state == "RUNNING" and not self.worker.is_busy:
                self.worker.frame = frame_rgb.copy()
                self.worker.percentile = getattr(self, 'percentile', 80)
                # 🚀 将 Widget 绑定的变量传给后台 worker
                self.worker.kernel_size = getattr(self, 'kernel_size', 5)
                self.worker.start()



    def _on_worker_finished(self, result_rgb):
        if self.state == "STOPPED":
            return
            
        h, w, c = result_rgb.shape
        qimg = QImage(result_rgb.data, w, h, c * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.lbl_thumbnail.width(), 
            self.lbl_thumbnail.height(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        self.lbl_thumbnail.setPixmap(pixmap)