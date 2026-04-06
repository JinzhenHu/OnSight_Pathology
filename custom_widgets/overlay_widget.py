import cv2
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame, QGraphicsDropShadowEffect
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QColor

from utils import find_clusters

# class OverlayWorker(QThread):
#     """Background thread to process the overlay."""
#     finished = pyqtSignal(np.ndarray)

#     def __init__(self):
#         super().__init__()
#         self.frame = None
#         self.is_busy = False

#     def run(self):
#         self.is_busy = True
#         try:
#             # tissue_mask = get_tissue_mask(self.frame)
#             # deconv, imH, imE = get_he_deconvolution(self.frame)
#             # score_map = compute_suspicious_score_map(self.frame, imH, imE, tissue_mask)
#             # result = visualize_hotspot_overlay(self.frame, score_map, tissue_mask)
#             _, _, result = find_hotspots_with_histomicstk(self.frame)
#             self.finished.emit(result)
#         except Exception as e:
#             print(f"[Overlay Error]: {e}")
#         finally:
#             self.is_busy = False

from torchvision import transforms
import torch
import timm
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
class OverlayWorker(QThread):
    """Background thread to process the overlay using DINOv2 clustering."""
    finished = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.frame = None
        self.is_busy = False

        # 1. Initialize Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing DINOv2 OverlayWorker on: {self.device}")

        # 2. Load Model ONCE
        # Using dinov2_vits14 as requested. 
        self.model = timm.create_model('mobilenetv3_small_100.lamb_in1k', pretrained=True, features_only=True, out_indices=(1, 2, 3, 4))
        #self.model = timm.create_model('tinynet_e.in1k', pretrained=True, num_classes=0)
        self.model.eval().to(self.device)

        # 3. Create Standard Transform
        # DINOv2 works best with inputs normalized to ImageNet standards
        self.config = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**self.config)

    def run(self):
            self.is_busy = True
            try:
                # Pass the pre-loaded model and transform to the utility function
                result = find_clusters(
                    img=self.frame, 
                    model=self.model, 
                    transform=self.transform, 
                    device=self.device,
                    patch_size=getattr(self, 'patch_size', 48),  # Dynamic
                    n_clusters=getattr(self, 'n_clusters', 5),   # Dynamic
                    # 🚀 [新增动态提取这三个参数]
                    sat_thresh=getattr(self, 'sat_thresh', 10), 
                    val_thresh=getattr(self, 'val_thresh', 250),
                    tissue_threshold=getattr(self, 'tissue_thresh', 0.95),
                    batch_size=128
                )
                cv2.imwrite("debug_overlay_cluster.png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR)) 
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

    # def process_frame(self, frame_rgb):
    #     if self.state == "RUNNING" and not self.worker.is_busy:
    #         self.worker.frame = frame_rgb.copy()
    #         self.worker.start()

    ############################################################################
    #clustering新代码
    ############################################################################
    def process_frame(self, frame_rgb):
            if self.state == "RUNNING" and not self.worker.is_busy:
                self.worker.frame = frame_rgb.copy()
                # Feed the dynamic parameters to the worker thread
                self.worker.patch_size = getattr(self, 'patch_size', 48)
                self.worker.n_clusters = getattr(self, 'n_clusters', 5)
                # 🚀 [将 UI 数据传递给 Worker]
                self.worker.sat_thresh = getattr(self, 'sat_thresh', 10)
                self.worker.val_thresh = getattr(self, 'val_thresh', 250)
                self.worker.tissue_thresh = getattr(self, 'tissue_thresh', 0.95)
                
                self.worker.start()
    ############################################################################
    #clustering新代码
    ############################################################################

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