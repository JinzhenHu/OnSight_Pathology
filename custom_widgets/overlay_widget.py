# Custom Overlay Widget for Clustering Visualization in OnSight Pathology
import cv2
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame, QGraphicsDropShadowEffect
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QColor
import os
from utils import find_clusters

from utils import resource_path
from torchvision import transforms
import torch
import timm
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
class OverlayWorker(QThread):
    """Background thread to process the overlay using clustering."""
    finished = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.frame = None
        self.is_busy = False

        # 1. Initialize Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(f"Initializing OverlayWorker on: {self.device}")
        #model_path = resource_path(os.path.join("local_models", "model.safetensors"))
        model_path = r"D:\UofT\2025fall\OnSight\OnSight_Pathology\local_models\model.safetensors"
        if os.path.exists(model_path):
            # print(f"Found local model at {model_path}. Loading offline...")
            self.model = timm.create_model(
                'mobilenetv3_small_100.lamb_in1k', 
                pretrained=False, 
                features_only=True, 
                out_indices=(1, 2, 3, 4)
            )
            from safetensors.torch import load_file
            state_dict = load_file(model_path)
            self.model.load_state_dict(state_dict, strict=False)
        else:
            #print(f"Local model not found. Downloading from HuggingFace...")
            self.model = timm.create_model(
                'mobilenetv3_small_100.lamb_in1k', 
                pretrained=True, 
                features_only=True, 
                out_indices=(1, 2, 3, 4)
            )
        self.model.eval().to(self.device)

        # 3. Create Standard Transform
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
                    patch_size=getattr(self, 'patch_size', 48), 
                    n_clusters=getattr(self, 'n_clusters', 5),   
                    sat_thresh=getattr(self, 'sat_thresh', 10), 
                    val_thresh=getattr(self, 'val_thresh', 250),
                    tissue_threshold=getattr(self, 'tissue_thresh', 0.95),
                    batch_size=128
                )
                #cv2.imwrite("debug_overlay_cluster.png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR)) 
                self.finished.emit(result)
            except Exception as e:
                print(f"[Overlay Error]: {e}")
            finally:
                self.is_busy = False

class OverlayWidget(QFrame):
    """The floating widget containing only the thumbnail image."""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Clean styling for a white background environment
        self.setStyleSheet("""
            OverlayWidget {
                background-color: rgba(255, 255, 255, 240); /* Slightly transparent white */
                border: 1px solid #dcdde1; /* Very soft silver/gray border */
                border-radius: 8px; /* Smooth rounded corners */
            }
        """)
        
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 40)) 
        shadow.setOffset(0, 4) 
        self.setGraphicsEffect(shadow)
        
        self.state = "STOPPED"
        self.worker = OverlayWorker()
        self.worker.finished.connect(self._on_worker_finished)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(4, 4, 4, 4)
        
        self.lbl_thumbnail = QLabel()
        self.lbl_thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
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


    ############################################################################
    #clustering
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
    #clustering
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