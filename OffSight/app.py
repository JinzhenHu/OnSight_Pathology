# PyQt6 interactive heatmap overlay viewer + slide runner (thumbnail + csv generator)
#
# Viewer expects a folder containing:
#   - thumbnail.jpg
#   - info_df_mitosis.csv and/or info_df_havoc.csv
#
# Run component:
#   - Select a folder containing slide files (.svs/.ndpi/.tif/.tiff/.jpg/.png/.jpeg)
#   - Generates per-slide output folders under an output root:
#       <output_root>/<slide_name>/{thumbnail.jpg, tiles/, info_df_mitosis.csv, info_df_havoc.csv}
#
# Run:
#   python main.py

from __future__ import annotations

import os
import sys

if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

import math
import re
import shutil
import pathlib
import json
import tempfile
import traceback
import multiprocessing as mp
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import cv2

from PyQt6.QtCore import (
    Qt,
    QObject,
    QProcess,
    QThread,
    pyqtSignal,
    QRunnable,
    QThreadPool,
    QTimer,
)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QLabel,
    QSlider,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGroupBox,
    QFormLayout,
    QDoubleSpinBox,
    QMessageBox,
    QCheckBox,
    QLineEdit,
    QProgressBar,
    QSizePolicy,
    QDialog,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QDialogButtonBox,
    QSpinBox,
    QTabWidget,
    QTextBrowser,
)


# ----------------------------
# Utilities / parsing
# ----------------------------

def parse_scores_str(x, lower=0.4, upper=1.0) -> np.ndarray:
    """Parse a comma-separated list of floats, filter finite, clamp to [lower, upper]."""
    if x is None:
        return np.array([], dtype=np.float32)
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return np.array([], dtype=np.float32)
        parts = [p.strip() for p in x.split(",") if p.strip() != ""]
        if not parts:
            return np.array([], dtype=np.float32)
        vals = np.array([float(p) for p in parts], dtype=np.float32)
    else:
        vals = np.asarray(x, dtype=np.float32)

    vals = vals[np.isfinite(vals)]
    vals = vals[(vals >= lower) & (vals <= upper)]
    return vals.astype(np.float32, copy=False)


def bgr_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    """Convert OpenCV BGR uint8 image to QPixmap."""
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError("Expected uint8 BGR image with shape (H,W,3).")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def is_openslide_like(img) -> bool:
    mod = type(img).__module__.lower()
    name = type(img).__name__.lower()
    return ("openslide" in mod) or ("openslide" in name)


APP_NAME = "onsight-wsi"
HF_SERVICE = "huggingface"
HF_REPO_ID = "prov-gigapath/prov-gigapath"
RUN_EVENT_PREFIX = "WSI_RUN_EVENT "
_MITOSIS_MODEL_CACHE = None
_HAVOC_FEATURE_EXTRACTOR_CACHE = None
_TILE_WORKER_SLIDE = None
_TILE_WORKER_MODIFIED_TILE = 0
_TILE_WORKER_TILE_SIZE = 0
_TILE_WORKER_FACTOR = 1.0
_TILE_WORKER_RESIZE_FACTOR = 1.0
_TILE_WORKER_MIN_TISSUE = 0.2
_TILE_WORKER_TILES_DIR = ""
_TILE_WORKER_PREVIEW_SIZE = 0


def get_hf_token() -> Optional[str]:
    import keyring

    return keyring.get_password(HF_SERVICE, APP_NAME)


def save_hf_token(token: str):
    import keyring

    keyring.set_password(HF_SERVICE, APP_NAME, token)


def login_huggingface(token: str):
    import huggingface_hub

    huggingface_hub.login(token=token, add_to_git_credential=False)


def validate_hf_token(token: str) -> tuple[bool, str]:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

    try:
        api = HfApi()
        api.whoami(token=token)
        api.model_info(repo_id=HF_REPO_ID, token=token)
        return True, "OK"
    except GatedRepoError:
        return False, "gated"
    except HfHubHTTPError:
        return False, "invalid"
    except Exception:
        return False, "network"

def is_havoc_model_downloaded(token: str) -> bool:
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(
            repo_id=HF_REPO_ID,
            token=token,
            local_files_only=True,
            allow_patterns=["config.json", "pytorch_model.bin"], # same as timm
        )
        return True
    except Exception:
        return False


def emit_run_event(event_type: str, **payload):
    msg = {"type": event_type, **payload}
    print(f"{RUN_EVENT_PREFIX}{json.dumps(msg)}", flush=True)


def _init_tile_worker(
    slide_path: str,
    modified_tile: int,
    tile_size: int,
    factor: float,
    resize_factor: float,
    min_tissue_amt: float,
    tiles_dir: str,
    preview_size: int,
):
    global _TILE_WORKER_SLIDE
    global _TILE_WORKER_MODIFIED_TILE
    global _TILE_WORKER_TILE_SIZE
    global _TILE_WORKER_FACTOR
    global _TILE_WORKER_RESIZE_FACTOR
    global _TILE_WORKER_MIN_TISSUE
    global _TILE_WORKER_TILES_DIR
    global _TILE_WORKER_PREVIEW_SIZE

    _TILE_WORKER_SLIDE = Slide(slide_path)
    _TILE_WORKER_MODIFIED_TILE = int(modified_tile)
    _TILE_WORKER_TILE_SIZE = int(tile_size)
    _TILE_WORKER_FACTOR = float(factor)
    _TILE_WORKER_RESIZE_FACTOR = float(resize_factor)
    _TILE_WORKER_MIN_TISSUE = float(min_tissue_amt)
    _TILE_WORKER_TILES_DIR = str(tiles_dir)
    _TILE_WORKER_PREVIEW_SIZE = int(preview_size)


def _tile_worker_read_region_bgr(x: int, y: int, w: int, h: int) -> np.ndarray:
    slide = _TILE_WORKER_SLIDE
    if slide is None:
        raise RuntimeError("Tile worker slide is not initialized.")

    if is_openslide_like(slide.image):
        rgba = np.array(slide.image.read_region((x, y), 0, (w, h)), dtype=np.uint8)
        rgb = rgba[:, :, :3]
        return rgb[:, :, ::-1].copy()

    crop = slide.image.crop((x, y, x + w, y + h)).convert("RGB")
    rgb = np.array(crop, dtype=np.uint8)
    return rgb[:, :, ::-1].copy()


def _process_tile_coord(coord: tuple[int, int]) -> tuple[tuple[int, int, int, int], np.ndarray, bool]:
    x0, y0 = coord
    macro_bgr = _tile_worker_read_region_bgr(x0, y0, _TILE_WORKER_MODIFIED_TILE, _TILE_WORKER_MODIFIED_TILE)
    if _TILE_WORKER_FACTOR != 1.0:
        macro_bgr = cv2.resize(
            macro_bgr,
            (_TILE_WORKER_TILE_SIZE, _TILE_WORKER_TILE_SIZE),
            interpolation=cv2.INTER_AREA,
        )

    base_out_x = int(x0 * _TILE_WORKER_RESIZE_FACTOR)
    base_out_y = int(y0 * _TILE_WORKER_RESIZE_FACTOR)
    coords = (
        base_out_x,
        base_out_y,
        base_out_x + _TILE_WORKER_TILE_SIZE,
        base_out_y + _TILE_WORKER_TILE_SIZE,
    )

    preview = cv2.resize(
        macro_bgr,
        (_TILE_WORKER_PREVIEW_SIZE, _TILE_WORKER_PREVIEW_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    keep_tile = amount_tissue(macro_bgr) >= _TILE_WORKER_MIN_TISSUE
    if keep_tile:
        tile_path = os.path.join(_TILE_WORKER_TILES_DIR, tile_coords_to_name(coords))
        if not cv2.imwrite(tile_path, macro_bgr):
            raise RuntimeError(f"Failed to save tile image: {tile_path}")

    return coords, preview, keep_tile


# ----------------------------
# Thumbnail builder (your efficient "free" approach)
# ----------------------------

class ImageCreator:
    """
    Creates an image of size (height/scale_factor, width/scale_factor).
    You "paint" tiles into it using output-space coordinates (x1,y1,x2,y2).
    """
    def __init__(self, height, width, scale_factor=1, channels=3):
        self.image = np.ones((int(height / scale_factor), int(width / scale_factor), channels), dtype=np.uint8) * 255
        self.scale_factor = scale_factor

    def _get_scaled_coordinate(self, coordinate):
        return tuple(int(c / self.scale_factor) for c in coordinate)

    def add_tile(self, tile_bgr: np.ndarray, coordinate):
        x1_adj, y1_adj, x2_adj, y2_adj = self._get_scaled_coordinate(coordinate)

        H, W = self.image.shape[:2]
        x1_adj = clamp_int(x1_adj, 0, W)
        x2_adj = clamp_int(x2_adj, 0, W)
        y1_adj = clamp_int(y1_adj, 0, H)
        y2_adj = clamp_int(y2_adj, 0, H)
        if x2_adj <= x1_adj or y2_adj <= y1_adj:
            return

        dst_w = x2_adj - x1_adj
        dst_h = y2_adj - y1_adj
        self.image[y1_adj:y2_adj, x1_adj:x2_adj, :] = cv2.resize(
            tile_bgr, (dst_w, dst_h), interpolation=cv2.INTER_AREA
        )


# ----------------------------
# Heatmap viewer data structs
# ----------------------------

@dataclass(frozen=True)
class RowDatum:
    x1: int
    y1: int
    x2: int
    y2: int
    vals_sorted: np.ndarray
    suffix_sum: np.ndarray

    def sum_ge(self, thr: float) -> float:
        if self.vals_sorted.size == 0:
            return 0.0
        idx = int(np.searchsorted(self.vals_sorted, thr, side="left"))
        if idx >= self.vals_sorted.size:
            return 0.0
        return float(self.suffix_sum[idx])


def build_heat_overlay_pixmap(
    thumb_bgr: np.ndarray,
    rows: List[RowDatum],
    scale_factor: int,
    score_threshold: float,
    blur_sigma: float,
) -> QPixmap:
    h_thumb, w_thumb = thumb_bgr.shape[:2]
    grid = np.zeros((h_thumb, w_thumb), dtype=np.float32)
    sf = float(scale_factor)

    for r in rows:
        # s = r.sum_ge(score_threshold)
        # if s <= 0.0:
        #     continue

        s = r.vals_sorted.max() if len(r.vals_sorted) else 0
        if s < score_threshold:
            continue



        x1 = int(r.x1 / sf)
        y1 = int(r.y1 / sf)
        x2 = int(r.x2 / sf)
        y2 = int(r.y2 / sf)

        x1 = clamp_int(x1, 0, w_thumb)
        x2 = clamp_int(x2, 0, w_thumb)
        y1 = clamp_int(y1, 0, h_thumb)
        y2 = clamp_int(y2, 0, h_thumb)
        if x2 <= x1 or y2 <= y1:
            continue

        grid[y1:y2, x1:x2] = s

    if blur_sigma > 0.0:
        grid = cv2.GaussianBlur(grid, ksize=(0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)

    # this combo makes the colorscale fixed
    minv = 0.4 # fixed. make the "white" color to always be assigned to 0.4
    maxv = float(grid.max()) # fixed never changing
    overlay_rgba = np.zeros((h_thumb, w_thumb, 4), dtype=np.uint8)
    if not math.isfinite(maxv) or maxv <= 0.0:
        qimg = QImage(overlay_rgba.data, w_thumb, h_thumb, 4 * w_thumb, QImage.Format.Format_RGBA8888)
        return QPixmap.fromImage(qimg.copy())

    # norm = np.clip(grid / maxv, 0.0, 1.0)
    # Normalize so the lowest non-zero value = white, highest = red
    norm = np.where(grid > 0.0, np.clip((grid - minv) / (maxv - minv + 1e-8), 0.0, 1.0), 0.0)

    overlay_rgba[..., 0] = 255
    overlay_rgba[..., 1] = np.clip(255.0 * (1.0 - norm), 0, 255).astype(np.uint8)
    overlay_rgba[..., 2] = np.clip(255.0 * (1.0 - norm), 0, 255).astype(np.uint8)
    overlay_rgba[..., 3] = np.where(norm > 0.0, 255, 0).astype(np.uint8)

    qimg = QImage(overlay_rgba.data, w_thumb, h_thumb, 4 * w_thumb, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


def build_havoc_overlay_pixmap(
    thumb_pixmap: QPixmap,
    df: Optional[pd.DataFrame],
    scale_factor: int,
    k_val: int,
) -> QPixmap:
    overlay = QPixmap(thumb_pixmap.size())
    overlay.fill(Qt.GlobalColor.transparent)
    if df is None or df.empty:
        return overlay

    r_col = f"k{k_val}_color_r"
    g_col = f"k{k_val}_color_g"
    b_col = f"k{k_val}_color_b"
    required = [r_col, g_col, b_col, *COORD_COLUMNS]
    if any(col not in df.columns for col in required):
        return overlay

    painter = QPainter(overlay)
    sf = float(scale_factor)
    for _, row in df.iterrows():
        x1 = int(row["coor_x1_20x"] / sf)
        y1 = int(row["coor_y1_20x"] / sf)
        x2 = int(row["coor_x2_20x"] / sf)
        y2 = int(row["coor_y2_20x"] / sf)
        if x2 <= x1 or y2 <= y1:
            continue

        r = float(row[r_col])
        g = float(row[g_col])
        b = float(row[b_col])
        if not np.isfinite([r, g, b]).all():
            continue
        if max(r, g, b) <= 1.0:
            r, g, b = r * 255.0, g * 255.0, b * 255.0

        pen = QPen(QColor(int(r), int(g), int(b)))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(x1, y1, max(1, x2 - x1 - 1), max(1, y2 - y1 - 1))

    painter.end()
    return overlay


# ----------------------------
# Viewer render task (threaded)
# ----------------------------

class RenderSignals(QObject):
    rendered = pyqtSignal(float, float, QPixmap)  # threshold, sigma, pixmap
    failed = pyqtSignal(str)


class RenderTask(QRunnable):
    def __init__(
        self,
        thumb_bgr: np.ndarray,
        rows: List[RowDatum],
        scale_factor: int,
        score_threshold: float,
        blur_sigma: float,
    ):
        super().__init__()
        self.signals = RenderSignals()
        self.thumb_bgr = thumb_bgr
        self.rows = rows
        self.scale_factor = int(scale_factor)
        self.score_threshold = float(score_threshold)
        self.blur_sigma = float(blur_sigma)

    def run(self):
        try:
            pm = build_heat_overlay_pixmap(
                self.thumb_bgr,
                self.rows,
                self.scale_factor,
                self.score_threshold,
                self.blur_sigma,
            )
            self.signals.rendered.emit(self.score_threshold, self.blur_sigma, pm)
        except Exception as e:
            self.signals.failed.emit(f"{type(e).__name__}: {e}")


class InteractiveGraphicsView(QGraphicsView):
    clicked = pyqtSignal(float, float)
    key_pressed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._zoom_level = 1.0
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def set_base_zoom(self):
        self._zoom_level = 1.0

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        scene_pos = self.mapToScene(event.pos())
        self.clicked.emit(scene_pos.x(), scene_pos.y())

    def wheelEvent(self, event):
        delta_y = event.angleDelta().y()
        if delta_y == 0:
            return

        zoom_factor = 1.15 if delta_y > 0 else 1 / 1.15
        next_zoom = self._zoom_level * zoom_factor
        if next_zoom < 1.0:
            zoom_factor = 1.0 / self._zoom_level
            self._zoom_level = 1.0
        else:
            self._zoom_level = next_zoom

        self.scale(zoom_factor, zoom_factor)
        event.accept()

    def keyPressEvent(self, event):
        key = event.key()
        if key in {Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down}:
            self.key_pressed.emit(key)
            event.accept()
            return
        super().keyPressEvent(event)


class ZoomableTilePreview(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = QPixmap()
        self._zoom = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0
        self._drag_active = False
        self._last_drag_pos = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def setPixmap(self, pixmap: QPixmap):
        self._pixmap = QPixmap(pixmap)
        self._zoom = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0
        self._drag_active = False
        self._last_drag_pos = None
        self.update()

    def clear(self):
        self._pixmap = QPixmap()
        self._zoom = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0
        self._drag_active = False
        self._last_drag_pos = None
        super().clear()
        self.update()

    def wheelEvent(self, event):
        if self._pixmap.isNull():
            super().wheelEvent(event)
            return

        old_zoom = self._zoom
        if event.angleDelta().y() > 0:
            self._zoom = min(self._zoom * 1.15, 12.0)
        elif event.angleDelta().y() < 0:
            self._zoom = max(self._zoom / 1.15, 1.0)
        else:
            return

        zoom_ratio = self._zoom / old_zoom if old_zoom > 0 else 1.0
        cx = event.position().x() - (self.width() / 2.0)
        cy = event.position().y() - (self.height() / 2.0)
        self._offset_x = (self._offset_x - cx) * zoom_ratio + cx
        self._offset_y = (self._offset_y - cy) * zoom_ratio + cy
        self._clamp_offsets()
        self.setCursor(Qt.CursorShape.OpenHandCursor if self._zoom > 1.0 else Qt.CursorShape.ArrowCursor)

        self.update()
        event.accept()

    def mousePressEvent(self, event):
        if (
            not self._pixmap.isNull()
            and self._zoom > 1.0
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._drag_active = True
            self._last_drag_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_active and self._last_drag_pos is not None:
            delta = event.position() - self._last_drag_pos
            self._offset_x += delta.x()
            self._offset_y += delta.y()
            self._last_drag_pos = event.position()
            self._clamp_offsets()
            self.update()
            event.accept()
            return
        if not self._pixmap.isNull():
            self.setCursor(Qt.CursorShape.OpenHandCursor if self._zoom > 1.0 else Qt.CursorShape.ArrowCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._drag_active and event.button() == Qt.MouseButton.LeftButton:
            self._drag_active = False
            self._last_drag_pos = None
            self.setCursor(
                Qt.CursorShape.OpenHandCursor if self._pixmap.isNull() is False and self._zoom > 1.0
                else Qt.CursorShape.ArrowCursor
            )
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def enterEvent(self, event):
        if not self._pixmap.isNull():
            self.setCursor(Qt.CursorShape.OpenHandCursor if self._zoom > 1.0 else Qt.CursorShape.ArrowCursor)
        super().enterEvent(event)

    def leaveEvent(self, event):
        if not self._drag_active:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().leaveEvent(event)

    def _clamp_offsets(self):
        if self._pixmap.isNull():
            self._offset_x = 0.0
            self._offset_y = 0.0
            return

        pm_w = self._pixmap.width()
        pm_h = self._pixmap.height()
        if pm_w <= 0 or pm_h <= 0:
            self._offset_x = 0.0
            self._offset_y = 0.0
            return

        base_scale = min(self.width() / pm_w, self.height() / pm_h)
        draw_w = pm_w * base_scale * self._zoom
        draw_h = pm_h * base_scale * self._zoom
        max_offset_x = max(0.0, (draw_w - self.width()) / 2.0)
        max_offset_y = max(0.0, (draw_h - self.height()) / 2.0)
        self._offset_x = max(-max_offset_x, min(max_offset_x, self._offset_x))
        self._offset_y = max(-max_offset_y, min(max_offset_y, self._offset_y))

    def paintEvent(self, event):
        if self._pixmap.isNull():
            super().paintEvent(event)
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        pm_w = self._pixmap.width()
        pm_h = self._pixmap.height()
        if pm_w <= 0 or pm_h <= 0:
            return

        base_scale = min(self.width() / pm_w, self.height() / pm_h)
        draw_w = pm_w * base_scale * self._zoom
        draw_h = pm_h * base_scale * self._zoom
        self._clamp_offsets()
        x = (self.width() - draw_w) / 2.0 + self._offset_x
        y = (self.height() - draw_h) / 2.0 + self._offset_y
        painter.drawPixmap(int(round(x)), int(round(y)), int(round(draw_w)), int(round(draw_h)), self._pixmap)


# ----------------------------
# MPP override dialog (table + quick set buttons)
# ----------------------------

class MppOverrideDialog(QDialog):
    """
    Shows a table of slides missing MPP and lets the user set per-slide MPP.
    Includes quick buttons: "Set all to 0.252" and "Set all to 0.504".
    """
    def __init__(self, missing_slide_paths: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Missing MPP values")
        self.setModal(True)
        self._paths = missing_slide_paths
        self._spinboxes: Dict[str, QDoubleSpinBox] = {}

        layout = QVBoxLayout()
        self.setLayout(layout)

        info = QLabel("These slides are missing MPP metadata.\n"
                      "Set MPP per slide below.\n"
                      "Note: 0.252 for 40x and 0.504 for 20x.")
        info.setWordWrap(True)
        layout.addWidget(info)

        # Quick buttons row
        quick_row = QHBoxLayout()
        btn_0252 = QPushButton("Set all to 0.252")
        btn_0504 = QPushButton("Set all to 0.504")
        btn_0252.clicked.connect(lambda: self.set_all(0.252))
        btn_0504.clicked.connect(lambda: self.set_all(0.504))
        btn_0252.setMinimumHeight(34)
        btn_0504.setMinimumHeight(34)
        quick_row.addWidget(btn_0252)
        quick_row.addWidget(btn_0504)
        quick_row.addStretch(1)
        layout.addLayout(quick_row)

        self.table = QTableWidget(len(missing_slide_paths), 2)
        self.table.setHorizontalHeaderLabels(["Slide", "MPP"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(260)

        for r, p in enumerate(missing_slide_paths):
            name = os.path.basename(p)
            item = QTableWidgetItem(name)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(r, 0, item)

            sb = QDoubleSpinBox()
            sb.setDecimals(3)
            sb.setRange(0.001, 10.0)
            sb.setSingleStep(0.001)
            sb.setValue(0.252)  # default
            sb.setMinimumHeight(30)
            sb.setFixedWidth(120)
            self.table.setCellWidget(r, 1, sb)
            self._spinboxes[p] = sb

        layout.addWidget(self.table)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.resize(700, 520)

    def set_all(self, mpp: float):
        for sb in self._spinboxes.values():
            sb.setValue(float(mpp))

    def get_overrides(self) -> Dict[str, float]:
        return {p: float(sb.value()) for p, sb in self._spinboxes.items()}


class HuggingFaceTokenDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HuggingFace Access")
        self.setModal(True)
        self.verified_token: Optional[str] = None

        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        self.setLayout(layout)

        # Title
        title = QLabel("Model Access Required")
        title_font = title.font()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        # Intro text
        intro = QLabel(
            "HAVOC uses the <b>Prov-GigaPath</b> pathology foundation model, which requires "
            "you to agree to its terms of use before it can be downloaded."
        )
        intro.setWordWrap(True)
        intro_font = intro.font()
        intro_font.setPointSize(10)
        intro.setFont(intro_font)
        layout.addWidget(intro)

        # Steps
        steps_label = QLabel("To get started:")
        steps_font = steps_label.font()
        steps_font.setPointSize(10)
        steps_font.setBold(True)
        steps_label.setFont(steps_font)
        layout.addWidget(steps_label)

        step_font = QFont()
        step_font.setPointSize(10)

        for step in [
            "1. Create a HuggingFace account and accept the model terms at:<br>"
            "&nbsp;&nbsp;&nbsp;&nbsp;<a href='https://huggingface.co/prov-gigapath/prov-gigapath'>"
            "huggingface.co/prov-gigapath/prov-gigapath</a>",

            "2. Generate a <b>read-only</b> user access token at:<br>"
            "&nbsp;&nbsp;&nbsp;&nbsp;<a href='https://huggingface.co/settings/tokens'>"
            "huggingface.co/settings/tokens</a>",

            "3. Paste your token below.",
        ]:
            step_label = QLabel(step)
            step_label.setFont(step_font)
            step_label.setOpenExternalLinks(True)
            step_label.setTextFormat(Qt.TextFormat.RichText)
            step_label.setWordWrap(True)
            layout.addWidget(step_label)

        # Footer note
        footer = QLabel("Your token will be stored securely and you will not be asked again.")
        footer_font = footer.font()
        footer_font.setPointSize(9)
        footer_font.setItalic(True)
        footer.setFont(footer_font)
        footer.setWordWrap(True)
        layout.addWidget(footer)

        layout.addSpacing(4)

        # Token input
        self.token_edit = QLineEdit()
        self.token_edit.setPlaceholderText("Paste your HuggingFace token here")
        self.token_edit.setEchoMode(QLineEdit.EchoMode.PasswordEchoOnEdit)
        self.token_edit.setMinimumHeight(36)
        layout.addWidget(self.token_edit)

        # Buttons
        button_row = QHBoxLayout()
        button_row.addStretch(1)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setMinimumHeight(36)
        cancel_btn.clicked.connect(self.reject)
        button_row.addWidget(cancel_btn)

        self.verify_btn = QPushButton("Verify Token")
        self.verify_btn.setMinimumHeight(36)
        self.verify_btn.setDefault(True)
        self.verify_btn.clicked.connect(self._verify_token)
        button_row.addWidget(self.verify_btn)

        layout.addLayout(button_row)
        self.resize(560, 400)

    def _verify_token(self):
        token = self.token_edit.text().strip()
        if not token:
            QMessageBox.warning(self, "Missing token", "Paste your HuggingFace token before verifying.")
            return

        self.verify_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            valid, reason = validate_hf_token(token)
            if valid:
                save_hf_token(token)
                login_huggingface(token)
        except Exception as e:
            QMessageBox.critical(self, "Verification failed", f"{type(e).__name__}: {e}")
            return
        finally:
            QApplication.restoreOverrideCursor()
            self.verify_btn.setEnabled(True)

        if valid:
            self.verified_token = token
            self.accept()
            return

        if reason == "gated":
            QMessageBox.critical(
                self,
                "Model Access Required",
                "You have not accepted the Prov-GigaPath terms of use. "
                "Please visit https://huggingface.co/prov-gigapath/prov-gigapath "
                "and accept the terms before continuing.",
            )
        elif reason == "invalid":
            QMessageBox.critical(self, "Invalid token", "Invalid token. Please check your token and try again.")
        else:
            QMessageBox.critical(
                self,
                "Connection error",
                "Could not connect to HuggingFace. Please check your internet connection.",
            )


class HuggingFaceDownloadWorker(QThread):
    status = pyqtSignal(str)
    download_finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, token: str, parent=None):
        super().__init__(parent)
        self.token = token

    def run(self):
        try:
            from huggingface_hub import snapshot_download
            self.status.emit("Downloading Prov-GigaPath model...")
            snapshot_download(
                repo_id=HF_REPO_ID,
                token=self.token,
                resume_download=True,
                allow_patterns=["config.json", "pytorch_model.bin"], # same as timm
            )
            self.status.emit("Download complete.")
            self.download_finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class HuggingFaceDownloadDialog(QDialog):
    def __init__(self, worker: HuggingFaceDownloadWorker, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Downloading HAVOC model")
        self.setModal(False)
        self.worker = worker

        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        self.setLayout(layout)

        # Title
        title = QLabel("Downloading Prov-GigaPath")
        title_font = title.font()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        # Info text
        info = QLabel(
            "The Prov-GigaPath model is approximately <b>5GB</b> and may take a while to download "
            "depending on your connection. You can close this window and the download will "
            "continue in the background."
        )
        info.setWordWrap(True)
        info_font = info.font()
        info_font.setPointSize(10)
        info.setFont(info_font)
        layout.addWidget(info)

        layout.addSpacing(4)

        # Status label
        self.status_label = QLabel("Starting download...")
        status_font = self.status_label.font()
        status_font.setPointSize(9)
        self.status_label.setFont(status_font)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setMinimumHeight(20)
        layout.addWidget(self.progress_bar)

        layout.addStretch(1)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.setMinimumHeight(36)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn, 0, Qt.AlignmentFlag.AlignRight)

        self.worker.status.connect(self.status_label.setText)
        self.worker.download_finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.resize(480, 240)

    def on_finished(self):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.status_label.setText("Download complete.")
        self.accept()

    def on_error(self, msg: str):
        self.close()


# ----------------------------
# Slide runner (thumbnail + csv)
# ----------------------------

SUPPORTED_SLIDE_EXTS = {".svs", ".ndpi", ".tif", ".tiff", ".jpg", ".jpeg", ".png"}
COORD_COLUMNS = ["coor_x1_20x", "coor_y1_20x", "coor_x2_20x", "coor_y2_20x"]
_TILE_COORD_RE = re.compile(r"^\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)$")


@dataclass(frozen=True)
class SavedTile:
    path: str
    coords: Tuple[int, int, int, int]


@dataclass(frozen=True)
class PendingRunConfig:
    slide_paths: List[str]
    mpp_overrides: Dict[str, float]
    output_root: str
    run_mitosis: bool
    run_havoc: bool
    havoc_tile_size: Optional[int]


def tile_coords_to_name(coords: Tuple[int, int, int, int]) -> str:
    return f"({coords[0]}, {coords[1]}, {coords[2]}, {coords[3]}).jpg"


def parse_tile_coords(tile_path: str) -> Tuple[int, int, int, int]:
    stem = pathlib.Path(tile_path).stem
    match = _TILE_COORD_RE.match(stem)
    if not match:
        raise ValueError(f"Tile filename does not encode coordinates: {tile_path}")
    return tuple(int(match.group(i)) for i in range(1, 5))


def load_mitosis_model():
    """
    Your loader (adjust import paths if needed).
    """
    global _MITOSIS_MODEL_CACHE
    if _MITOSIS_MODEL_CACHE is not None:
        return _MITOSIS_MODEL_CACHE

    import yaml
    from huggingface_hub import hf_hub_download
    from retinanet.Model import MyMitosisDetection  # <-- adjust if your package path differs

    # Always use this for accessing any local path
    def resource_path(relative_path):
        """Get absolute path to resource (for dev and for PyInstaller onefile mode)"""
        if hasattr(sys, '_MEIPASS'):
            # _MEIPASS is the temp folder where PyInstaller unpacks files
            return os.path.join(sys._MEIPASS, relative_path)
        return os.path.join(os.path.abspath("."), relative_path)

    REPO = "diamandislabii/Mitosis-detection"
    model_path = hf_hub_download(repo_id=REPO, filename="bestmodel.pth")
    config_path = resource_path("retinanet/file/config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    detector = MyMitosisDetection(model_path, config)
    detector.load_model()
    _MITOSIS_MODEL_CACHE = detector
    return _MITOSIS_MODEL_CACHE


def load_havoc_runner(tile_size: int):
    global _HAVOC_FEATURE_EXTRACTOR_CACHE

    from havoc.havoc import HAVOC, HAVOCConfig

    if _HAVOC_FEATURE_EXTRACTOR_CACHE is None:
        from havoc.feature_extractor import FeatureExtractor

        _HAVOC_FEATURE_EXTRACTOR_CACHE = FeatureExtractor()

    return HAVOC(HAVOCConfig(tile_size=tile_size), feature_extractor=_HAVOC_FEATURE_EXTRACTOR_CACHE)


class SlideLoaderError(Exception):
    pass


class Slide:
    """
    Minimal metadata loader for mpp.
    Lazy imports so viewer can run without openslide/pillow installed.
    """
    def __init__(self, path: str, mpp: Optional[float] = None):
        self.path = path
        self.name = pathlib.Path(path).stem
        self.image_type = pathlib.Path(path).suffix.lower()

        if self.image_type not in SUPPORTED_SLIDE_EXTS:
            raise SlideLoaderError(f"Unsupported image type: {self.image_type}")

        if self.image_type in {".svs", ".ndpi", ".tif", ".tiff"}:
            try:
                import openslide  # noqa
            except Exception as e:
                raise SlideLoaderError(
                    "openslide-python is required to open SVS/NDPI/TIF. "
                    f"Import failed: {type(e).__name__}: {e}"
                )
            import openslide
            img = openslide.open_slide(self.path)
            w, h = img.dimensions
        else:
            try:
                from PIL import Image, ImageFile
                Image.MAX_IMAGE_PIXELS = 100000000000
                ImageFile.LOAD_TRUNCATED_IMAGES = True
            except Exception as e:
                raise SlideLoaderError(
                    "Pillow is required to open JPG/PNG. "
                    f"Import failed: {type(e).__name__}: {e}"
                )
            from PIL import Image
            img = Image.open(self.path)
            w, h = img.width, img.height

        self.width = int(w)
        self.height = int(h)
        self.image = img

        if self.image_type == ".svs":
            curr = self._extract_data_svs()
        elif self.image_type == ".ndpi":
            curr = self._extract_data_ndpi()
        else:
            curr = {"mpp": None}

        self.mpp = float(mpp) if mpp is not None else curr.get("mpp", None)

    def _extract_data_svs(self) -> dict:
        props = getattr(self.image, "properties", {})
        if "tiff.ImageDescription" in props:
            desc = props["tiff.ImageDescription"]
            mpp = re.search(r"MPP\s*=\s*([\d.]+)", desc)
            return {"mpp": float(mpp.group(1)) if mpp else None}
        return {"mpp": None}

    def _extract_data_ndpi(self) -> dict:
        props = getattr(self.image, "properties", {})
        if "openslide.mpp-x" in props:
            try:
                return {"mpp": float(props["openslide.mpp-x"])}
            except Exception:
                return {"mpp": None}
        return {"mpp": None}


class SlideRunSignals(QObject):
    started = pyqtSignal(int)          # num slides
    progress = pyqtSignal(str)         # status string
    step_progress = pyqtSignal(int)    # percentage for current slide (0-100)
    slide_done = pyqtSignal(str, str)  # slide name, output folder
    finished = pyqtSignal(str)         # output root
    failed = pyqtSignal(str)


def amount_tissue(tile_bgr: np.ndarray) -> float:
    score1 = 1 - (np.sum(np.std(tile_bgr, axis=2) < 2.75) / (tile_bgr.shape[0] * tile_bgr.shape[1]))
    bw = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    score2 = 1 - ((bw > 220).sum() / (bw.shape[0] * bw.shape[1]))
    return float(min(score1, score2))


def _scores_to_str(scores) -> str:
    if scores is None:
        return ""
    if isinstance(scores, (float, int, np.floating, np.integer)):
        return f"{float(scores):.2f}"
    arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return ""
    return ",".join([f"{float(x):.2f}" for x in arr])


class SlideRunTask(QRunnable):
    """
    Generates thumbnail.jpg and per-pipeline info_df.csv outputs for all slides.
    """
    def __init__(
        self,
        slide_paths: List[str],
        mpp_overrides: Dict[str, float],
        output_root: str,
        run_mitosis: bool,
        run_havoc: bool,
        havoc_tile_size: Optional[int] = None,
    ):
        super().__init__()
        self.setAutoDelete(False)
        self.signals = SlideRunSignals()
        self.slide_paths = slide_paths
        self.mpp_overrides = dict(mpp_overrides)
        self.output_root = output_root
        self.run_mitosis = bool(run_mitosis)
        self.run_havoc = bool(run_havoc)

        self.desired_tile_mpp = 0.504  # 20x
        self.havoc_tile_size = int(havoc_tile_size) if havoc_tile_size is not None else 512
        self.tile_size = self.havoc_tile_size if self.run_havoc else 256
        self.map_scale_fac = 32
        self.min_tissue_amt = 0.2

    def run(self):
        try:
            os.makedirs(self.output_root, exist_ok=True)

            mitosis_model = None
            havoc_runner = None

            if self.run_mitosis:
                self.signals.progress.emit("Loading mitosis model...")
                mitosis_model = load_mitosis_model()

            if self.run_havoc:
                self.signals.progress.emit("Loading HAVOC feature extractor...")
                havoc_runner = load_havoc_runner(self.havoc_tile_size)

            self.signals.started.emit(len(self.slide_paths))

            for si, sp in enumerate(self.slide_paths, start=1):
                slide_name = pathlib.Path(sp).stem
                self.signals.progress.emit(f"[{si}/{len(self.slide_paths)}] Opening {slide_name}...")

                slide = Slide(sp)
                if slide.mpp is None:
                    if sp not in self.mpp_overrides:
                        raise SlideLoaderError(f"MPP missing for {slide_name} and no override provided.")
                    slide = Slide(sp, mpp=self.mpp_overrides[sp])

                slide_out = os.path.join(self.output_root, slide_name)
                os.makedirs(slide_out, exist_ok=True)

                if not self.run_mitosis:
                    mitosis_csv = os.path.join(slide_out, "info_df_mitosis.csv")
                    if os.path.exists(mitosis_csv):
                        os.remove(mitosis_csv)
                if not self.run_havoc:
                    havoc_csv = os.path.join(slide_out, "info_df_havoc.csv")
                    if os.path.exists(havoc_csv):
                        os.remove(havoc_csv)

                saved_tiles = self._tile_slide(slide, slide_out, slide_name, si)

                if self.run_havoc and havoc_runner is not None:
                    self.signals.progress.emit(f"[{si}/{len(self.slide_paths)}] {slide_name}: HAVOC...")
                    havoc_df = havoc_runner.run_tiles(
                        os.path.join(slide_out, "tiles"),
                        progress_callback=self.signals.progress.emit,
                        slide_name=slide_name,
                    )
                    havoc_df.to_csv(os.path.join(slide_out, "info_df_havoc.csv"), index=False)

                if self.run_mitosis and mitosis_model is not None:
                    self.signals.progress.emit(f"[{si}/{len(self.slide_paths)}] {slide_name}: Mitosis...")
                    mitosis_df = self._run_mitosis_on_saved_tiles(saved_tiles, mitosis_model, slide_name)
                    mitosis_df.to_csv(os.path.join(slide_out, "info_df_mitosis.csv"), index=False)

                self.signals.slide_done.emit(slide_name, slide_out)

            self.signals.finished.emit(self.output_root)

        except Exception as e:
            self.signals.failed.emit(f"{type(e).__name__}: {e}")

    def _tile_slide(self, slide: Slide, slide_out: str, slide_name: str, slide_idx: int) -> List[SavedTile]:
        slide_mpp = float(slide.mpp)
        if abs(slide_mpp - self.desired_tile_mpp) < 0.01:
            slide_mpp = self.desired_tile_mpp

        factor = self.desired_tile_mpp / round(slide_mpp, 3) if slide_mpp else 1.0
        modified_tile = max(1, int(round(self.tile_size * factor)))
        trimmed_w = slide.width - (slide.width % modified_tile)
        trimmed_h = slide.height - (slide.height % modified_tile)

        resize_factor = 1.0 / factor
        out_w = int(trimmed_w * resize_factor)
        out_h = int(trimmed_h * resize_factor)

        tiles_dir = os.path.join(slide_out, "tiles")
        if os.path.isdir(tiles_dir):
            shutil.rmtree(tiles_dir)
        os.makedirs(tiles_dir, exist_ok=True)

        extraction_map = ImageCreator(
            height=out_h,
            width=out_w,
            scale_factor=self.map_scale_fac,
            channels=3,
        )

        saved_tiles: List[SavedTile] = []
        num_macro_x = trimmed_w // modified_tile if modified_tile else 0
        num_macro_y = trimmed_h // modified_tile if modified_tile else 0
        total_macro = max(1, num_macro_x * num_macro_y)
        preview_size = max(1, self.tile_size // self.map_scale_fac)

        self.signals.progress.emit(f"[{slide_idx}/{len(self.slide_paths)}] {slide_name}: tiling...")
        coords_list = [
            (x0, y0)
            for y0 in range(0, trimmed_h, modified_tile)
            for x0 in range(0, trimmed_w, modified_tile)
        ]

        max_workers = min(8, max(1, mp.cpu_count()), len(coords_list))
        if not is_openslide_like(slide.image):
            max_workers = 1
        report_every = max(1, min(32, total_macro // 100 if total_macro > 100 else 1))

        if max_workers <= 1 or len(coords_list) <= 4:
            done_macro = 0
            _init_tile_worker(
                slide.path,
                modified_tile,
                self.tile_size,
                factor,
                resize_factor,
                self.min_tissue_amt,
                tiles_dir,
                preview_size,
            )
            for coord in coords_list:
                coords, preview_bgr, keep_tile = _process_tile_coord(coord)
                extraction_map.add_tile(preview_bgr, coords)
                if keep_tile:
                    saved_tiles.append(SavedTile(path=os.path.join(tiles_dir, tile_coords_to_name(coords)), coords=coords))

                done_macro += 1
                if done_macro == 1 or done_macro == total_macro or done_macro % report_every == 0:
                    pct = (done_macro / total_macro) * 100.0
                    self.signals.step_progress.emit(int(pct))
                    self.signals.progress.emit(f"{slide_name}: tiling {pct:.1f}%")
        else:
            ctx = mp.get_context("spawn")
            chunksize = max(1, min(16, len(coords_list) // (max_workers * 4) if len(coords_list) > max_workers else 1))
            done_macro = 0

            with ctx.Pool(
                processes=max_workers,
                initializer=_init_tile_worker,
                initargs=(
                    slide.path,
                    modified_tile,
                    self.tile_size,
                    factor,
                    resize_factor,
                    self.min_tissue_amt,
                    tiles_dir,
                    preview_size,
                ),
            ) as pool:
                for coords, preview_bgr, keep_tile in pool.imap_unordered(_process_tile_coord, coords_list, chunksize=chunksize):
                    extraction_map.add_tile(preview_bgr, coords)
                    if keep_tile:
                        saved_tiles.append(
                            SavedTile(path=os.path.join(tiles_dir, tile_coords_to_name(coords)), coords=coords)
                        )

                    done_macro += 1
                    if done_macro == 1 or done_macro == total_macro or done_macro % report_every == 0:
                        pct = (done_macro / total_macro) * 100.0
                        self.signals.step_progress.emit(int(pct))
                        self.signals.progress.emit(f"{slide_name}: tiling {pct:.1f}%")

        saved_tiles.sort(key=lambda tile: tile.coords)

        self.signals.step_progress.emit(100)
        self.signals.progress.emit(f"[{slide_idx}/{len(self.slide_paths)}] {slide_name}: tiling 100.0%")

        if not cv2.imwrite(os.path.join(slide_out, "thumbnail.jpg"), extraction_map.image):
            raise RuntimeError(f"Failed to save thumbnail for {slide_name}")

        return saved_tiles

    def _run_mitosis_on_saved_tiles(self, saved_tiles: List[SavedTile], model, slide_name: str) -> pd.DataFrame:
        rows = []
        total_tiles = max(1, len(saved_tiles))
        patches_per_row = max(1, self.tile_size // 256)

        for idx, saved_tile in enumerate(saved_tiles, start=1):
            if idx == 1 or idx == total_tiles or idx % 10 == 0:
                pct = (idx / total_tiles) * 100.0 if saved_tiles else 100.0
                self.signals.progress.emit(f"{slide_name}: mitosis {pct:.1f}%")

            tile_bgr = cv2.imread(saved_tile.path, cv2.IMREAD_COLOR)
            if tile_bgr is None:
                raise RuntimeError(f"Failed to read saved tile: {saved_tile.path}")

            tile_40x_bgr = cv2.resize(tile_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            tile_40x_rgb = cv2.cvtColor(tile_40x_bgr, cv2.COLOR_BGR2RGB)

            batch = []
            offsets = []
            tile_size_40x = 512
            for y0 in range(0, tile_40x_rgb.shape[0], tile_size_40x):
                for x0 in range(0, tile_40x_rgb.shape[1], tile_size_40x):
                    batch.append(tile_40x_rgb[y0:y0 + tile_size_40x, x0:x0 + tile_size_40x, :])
                    offsets.append((x0, y0))

            per_tile_scores, _detections_per_tile = model.run_on_tiles(batch)
            expected_scores = patches_per_row * patches_per_row
            if len(per_tile_scores) != expected_scores:
                raise RuntimeError(
                    f"Expected {expected_scores} mitosis scores for {saved_tile.path}, got {len(per_tile_scores)}."
                )

            def draw_detections_on_big_tile(big_tile_bgr, detections_per_tile, offsets, scores,
                                            score_min=0.4, thickness=7):
                out = big_tile_bgr.copy()
                H, W = out.shape[:2]

                for patch_idx, (x0, y0) in enumerate(offsets):
                    if scores[patch_idx] == [0]: continue # wont have detections so line below would crash

                    # go through all the detections per patch
                    for score, (x1, y1, x2, y2) in zip(scores[patch_idx], detections_per_tile[patch_idx]):

                        if score < score_min:
                            continue

                        # shift to big-tile coordinates
                        x1g = x1 + x0
                        x2g = x2 + x0
                        y1g = y1 + y0
                        y2g = y2 + y0

                        # clip
                        x1g = int(max(0, min(W - 1, round(x1g))))
                        y1g = int(max(0, min(H - 1, round(y1g))))
                        x2g = int(max(0, min(W - 1, round(x2g))))
                        y2g = int(max(0, min(H - 1, round(y2g))))

                        if score > 0.7:
                            box_color = (0, 255, 0)  # Green for high confidence
                        elif score > 0.55:
                            box_color = (255, 165, 0)  # Yellow for medium confidence
                        elif score > 0.4:
                            box_color = (0, 0, 255)  # Red for low confidence
                        else:
                            box_color = (0, 0, 0)

                        cv2.rectangle(out, (x1g, y1g), (x2g, y2g), box_color, thickness)

                        ###########

                        label = f'{(score * 100):.0f}%'
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1
                        font_thickness = 2
                        bg_color = (255, 255, 255)  # white background
                        pad = 5

                        (text_w, text_h), baseline = cv2.getTextSize(
                            label, font, font_scale, font_thickness
                        )

                        # Try to put text above the box
                        text_x = x1g
                        text_y = y1g - 10

                        # If it would go above the image, put it below the box
                        if text_y - text_h - baseline < 0:
                            text_y = y2g + text_h + baseline + 8

                        # Clamp X so text doesn't overflow right edge
                        if text_x + text_w > W:
                            text_x = W - text_w - 1

                        # Background rectangle coords
                        bg_x1 = max(0, text_x - pad)
                        bg_y1 = max(0, text_y - text_h - pad)
                        bg_x2 = min(W - 1, text_x + text_w + pad)
                        bg_y2 = min(H - 1, text_y + baseline + pad)

                        # Draw filled white rectangle
                        cv2.rectangle(out, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)

                        cv2.putText(
                            out,
                            label,
                            (text_x, text_y),
                            font,
                            font_scale,
                            box_color,
                            font_thickness,
                            cv2.LINE_AA
                        )

                        # cv2.putText(out, f'{(score * 100):.0f}%', (x1g, max(0, y1g - 10)),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 3, cv2.LINE_AA)

                return out

            if any(len(det) for det in _detections_per_tile):
                overlay = draw_detections_on_big_tile(tile_40x_bgr, _detections_per_tile, offsets, per_tile_scores, score_min=0.4)
                overlay = cv2.resize(overlay, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(saved_tile.path, overlay)

            for y0 in range(0, self.tile_size, 256):
                for x0 in range(0, self.tile_size, 256):
                    patch_index = (y0 // 256) * patches_per_row + (x0 // 256)
                    temp_x = saved_tile.coords[0] + x0
                    temp_y = saved_tile.coords[1] + y0
                    rows.append(
                        {
                            COORD_COLUMNS[0]: temp_x,
                            COORD_COLUMNS[1]: temp_y,
                            COORD_COLUMNS[2]: temp_x + 256,
                            COORD_COLUMNS[3]: temp_y + 256,
                            "mitosis_raw": _scores_to_str([per_tile_scores[patch_index]]),
                        }
                    )

        if not rows:
            return pd.DataFrame(columns=COORD_COLUMNS + ["mitosis_raw"])

        return pd.DataFrame(rows, columns=COORD_COLUMNS + ["mitosis_raw"])

    def _read_region_bgr(self, slide: Slide, x: int, y: int, w: int, h: int) -> np.ndarray:
        if is_openslide_like(slide.image):
            rgba = np.array(slide.image.read_region((x, y), 0, (w, h)), dtype=np.uint8)
            rgb = rgba[:, :, :3]
            return rgb[:, :, ::-1].copy()
        from PIL import Image
        crop = slide.image.crop((x, y, x + w, y + h)).convert("RGB")
        rgb = np.array(crop, dtype=np.uint8)
        return rgb[:, :, ::-1].copy()


def run_job_from_config(config_path: str) -> int:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    task = SlideRunTask(
        slide_paths=list(config["slide_paths"]),
        mpp_overrides=dict(config["mpp_overrides"]),
        output_root=str(config["output_root"]),
        run_mitosis=bool(config["run_mitosis"]),
        run_havoc=bool(config["run_havoc"]),
        havoc_tile_size=config.get("havoc_tile_size"),
    )

    state = {"finished": False, "failed": False}

    task.signals.started.connect(lambda n: emit_run_event("started", n=int(n)))
    task.signals.progress.connect(lambda msg: emit_run_event("progress", msg=str(msg)))
    task.signals.slide_done.connect(
        lambda slide_name, out_dir: emit_run_event("slide_done", slide_name=str(slide_name), out_dir=str(out_dir))
    )

    def on_finished(out_root: str):
        state["finished"] = True
        emit_run_event("finished", out_root=str(out_root))

    def on_failed(msg: str):
        state["failed"] = True
        emit_run_event("failed", msg=str(msg))

    task.signals.finished.connect(on_finished)
    task.signals.failed.connect(on_failed)

    task.run()

    if state["finished"] and not state["failed"]:
        return 0
    if not state["failed"]:
        emit_run_event("failed", msg="Run ended unexpectedly.")
    return 1


# ----------------------------
# Main window
# ----------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OffSight")

        font = self.font()
        font.setPointSize(12)
        self.setFont(font)

        self.thread_pool = QThreadPool.globalInstance()
        self.run_thread_pool = QThreadPool(self)
        self.run_thread_pool.setMaxThreadCount(1)
        self.run_thread_pool.setExpiryTimeout(-1)

        # Viewer state
        self.thumb_bgr: Optional[np.ndarray] = None
        self.thumb_pixmap: Optional[QPixmap] = None
        self.mitosis_rows: List[RowDatum] = []
        self.havoc_df: Optional[pd.DataFrame] = None
        self.saved_tiles: List[SavedTile] = []
        self.saved_tile_map: Dict[Tuple[int, int], SavedTile] = {}
        self.current_saved_tile: Optional[SavedTile] = None
        self.selection_rect_item: Optional[QGraphicsRectItem] = None
        self.viewer_tile_size: int = 512
        self.scale_factor: int = 32

        self.mitosis_overlay_cache: Dict[Tuple[float, float], QPixmap] = {}
        self.mitosis_cache_order: List[Tuple[float, float]] = []
        self.mitosis_cache_max_items = 24
        self.havoc_overlay_cache: Dict[int, QPixmap] = {}
        self.pending_run_config: Optional[PendingRunConfig] = None
        self.hf_download_worker: Optional[HuggingFaceDownloadWorker] = None
        self.hf_download_dialog: Optional[HuggingFaceDownloadDialog] = None
        self.run_process: Optional[QProcess] = None
        self.run_process_config_path: Optional[str] = None
        self.run_cancel_mode = False
        self.run_cancel_requested = False
        self.run_stdout_buffer = ""
        self.run_stderr_buffer = ""
        self.run_completed_out_root: Optional[str] = None
        self.run_failed_message: Optional[str] = None

        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self._kick_mitosis_render)

        # Graphics view
        self.scene = QGraphicsScene(self)
        self.view = InteractiveGraphicsView(self)
        self.view.setScene(self.scene)
        self.view.setRenderHints(
            self.view.renderHints()
            | QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.view.clicked.connect(self._viewer_clicked)
        self.view.key_pressed.connect(self._viewer_key_pressed)

        self.thumb_item = QGraphicsPixmapItem()
        self.havoc_overlay_item = QGraphicsPixmapItem()
        self.mitosis_overlay_item = QGraphicsPixmapItem()
        self.scene.addItem(self.thumb_item)
        self.scene.addItem(self.havoc_overlay_item)
        self.scene.addItem(self.mitosis_overlay_item)

        self.tile_preview_label = ZoomableTilePreview()
        self.tile_preview_label.setFixedSize(256, 256)
        self.tile_preview_label.setStyleSheet("border: 2px solid white; background: #111;")
        self.tile_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tile_preview_label.setText("Selected Tile")

        # ---------- Controls (viewer) ----------
        open_btn = QPushButton("Select results folder...")
        open_btn.setMinimumHeight(38)
        open_btn.clicked.connect(self.open_heatmap_folder)

        self.havoc_box = QGroupBox("HAVOC")
        self.havoc_box.setCheckable(True)
        self.havoc_box.setChecked(True)
        self.havoc_box.toggled.connect(self._update_havoc_overlay_visibility)

        self.havoc_k_slider = QSlider(Qt.Orientation.Horizontal)
        self.havoc_k_slider.setRange(2, 9)
        self.havoc_k_slider.setValue(4)
        self.havoc_k_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.havoc_k_slider.setTickInterval(1)
        self.havoc_k_slider.valueChanged.connect(self._havoc_k_changed)

        self.havoc_k_label = QLabel("k = 4")

        self.havoc_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.havoc_opacity_slider.setRange(0, 100)
        self.havoc_opacity_slider.setValue(35)
        self.havoc_opacity_slider.valueChanged.connect(self._havoc_opacity_changed)

        havoc_form = QFormLayout()
        havoc_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        havoc_form.addRow("Clusters:", self._hbox(self.havoc_k_slider, self.havoc_k_label))
        havoc_form.addRow("Overlay opacity:", self.havoc_opacity_slider)
        self.havoc_box.setLayout(havoc_form)

        self.mitosis_box = QGroupBox("Mitosis")
        self.mitosis_box.setCheckable(True)
        self.mitosis_box.setChecked(True)
        self.mitosis_box.toggled.connect(self._update_mitosis_overlay_visibility)

        self.thr_spin = QDoubleSpinBox()
        self.thr_spin.setDecimals(2)
        self.thr_spin.setSingleStep(0.01)
        self.thr_spin.setRange(0.40, 1.00)
        self.thr_spin.setValue(0.60)
        self.thr_spin.setMinimumHeight(32)
        self.thr_spin.setFixedWidth(100)

        self.thr_slider = QSlider(Qt.Orientation.Horizontal)
        self.thr_slider.setRange(40, 100)
        self.thr_slider.setValue(60)
        self.thr_slider.setMinimumHeight(32)
        self.thr_slider.valueChanged.connect(self._thr_slider_changed)
        self.thr_spin.valueChanged.connect(self._thr_spin_changed)

        self.mitosis_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.mitosis_opacity_slider.setRange(0, 100)
        self.mitosis_opacity_slider.setValue(85)
        self.mitosis_opacity_slider.setMinimumHeight(32)
        self.mitosis_opacity_slider.valueChanged.connect(self._mitosis_opacity_changed)

        # self.blur_check = QCheckBox("Gaussian blur")
        # self.blur_check.setChecked(False)

        # self.blur_sigma_spin = QDoubleSpinBox()
        # self.blur_sigma_spin.setDecimals(1)
        # self.blur_sigma_spin.setSingleStep(0.5)
        # self.blur_sigma_spin.setRange(0.0, 50.0)
        # self.blur_sigma_spin.setValue(7.0)
        # self.blur_sigma_spin.setMinimumHeight(32)
        # self.blur_sigma_spin.setFixedWidth(100)
        # self.blur_check.stateChanged.connect(self.request_mitosis_render)
        # self.blur_sigma_spin.valueChanged.connect(self.request_mitosis_render)

        controls = QGroupBox("")
        controls.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        mitosis_form = QFormLayout()
        mitosis_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        mitosis_form.addRow("Score threshold:", self._hbox(self.thr_slider, self.thr_spin))
        mitosis_form.addRow("Overlay opacity:", self.mitosis_opacity_slider)
        # mitosis_form.addRow(self.blur_check, self.blur_sigma_spin)
        self.mitosis_box.setLayout(mitosis_form)

        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(8, 8, 8, 8)
        controls_layout.setSpacing(14)
        controls_layout.addWidget(open_btn)
        controls_layout.addWidget(self.havoc_box)
        controls_layout.addWidget(self.mitosis_box)
        controls.setLayout(controls_layout)

        self.havoc_box.hide()
        self.mitosis_box.hide()

        # ---------- Run ----------
        run_box = QGroupBox("")
        run_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        self.slides_folder_edit = QLineEdit()
        self.slides_folder_edit.setReadOnly(True)
        self.slides_folder_edit.setMinimumHeight(32)

        self.browse_slides_btn = QPushButton("Browse...")
        self.browse_slides_btn.setMinimumHeight(34)
        self.browse_slides_btn.clicked.connect(self.browse_slides_folder)

        self.output_root_edit = QLineEdit()
        self.output_root_edit.setReadOnly(True)
        self.output_root_edit.setMinimumHeight(32)

        # self.browse_out_btn = QPushButton("Browse...")
        # self.browse_out_btn.setMinimumHeight(34)
        # self.browse_out_btn.setEnabled(False)
        # self.browse_out_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # self.browse_out_btn.setStyleSheet("color: transparent; background: transparent; border: none;")
        self.browse_out_btn = QWidget()
        self.browse_out_btn.setFixedWidth(self.browse_slides_btn.sizeHint().width())

        # greys them out
        self.slides_folder_edit.setStyleSheet("QLineEdit { background-color: palette(window); color: palette(mid); }")
        self.output_root_edit.setStyleSheet("QLineEdit { background-color: palette(window); color: palette(mid); }")

        self.mitosis_checkbox = QCheckBox("Mitosis")
        self.mitosis_checkbox.setChecked(True)
        self.mitosis_checkbox.toggled.connect(self._update_mitosis_tile_size_visibility)

        self.mitosis_tile_size_label = QLabel("Tile size:")
        self.mitosis_tile_size_spin = QSpinBox()
        self.mitosis_tile_size_spin.setRange(256, 256)
        self.mitosis_tile_size_spin.setSingleStep(256)
        self.mitosis_tile_size_spin.setValue(256)
        self.mitosis_tile_size_spin.setMinimumHeight(32)
        self.mitosis_tile_size_spin.setMinimumWidth(88)
        self.mitosis_tile_size_spin.setEnabled(False)

        self.havoc_checkbox = QCheckBox("HAVOC")
        self.havoc_checkbox.setChecked(True)
        self.havoc_checkbox.toggled.connect(self._update_havoc_tile_size_visibility)

        self.havoc_tile_size_label = QLabel("Tile size:")
        self.havoc_tile_size_spin = QSpinBox()
        self.havoc_tile_size_spin.setRange(256, 2048)
        self.havoc_tile_size_spin.setSingleStep(256)
        self.havoc_tile_size_spin.setValue(512)
        self.havoc_tile_size_spin.setMinimumHeight(32)
        self.havoc_tile_size_spin.setMinimumWidth(88)

        self.run_btn = QPushButton("Run slide processing")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.clicked.connect(self._handle_run_button_clicked)

        self.run_progress = QProgressBar()
        self.run_progress.setRange(0, 100)
        self.run_progress.setValue(0)
        self.run_progress.setMinimumHeight(26)

        self.run_status_label = QLabel("")
        self.run_status_label.setWordWrap(True)

        run_form = QFormLayout()
        run_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        run_form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        run_form.setVerticalSpacing(10)

        pipeline_widget = QWidget()
        pipeline_layout = QVBoxLayout()
        pipeline_layout.setContentsMargins(0, 0, 0, 0)
        pipeline_layout.setSpacing(8)
        pipeline_layout.addWidget(self._hbox(self.mitosis_checkbox, self.mitosis_tile_size_label, self.mitosis_tile_size_spin))
        pipeline_layout.addWidget(self._hbox(self.havoc_checkbox, self.havoc_tile_size_label, self.havoc_tile_size_spin))
        pipeline_widget.setLayout(pipeline_layout)

        run_form.addRow("Slides folder:", self._hbox(self.slides_folder_edit, self.browse_slides_btn))
        run_form.addRow("Output root:", self._hbox(self.output_root_edit, self.browse_out_btn))
        run_form.addRow("Pipelines:", pipeline_widget)
        run_form.addRow(self.run_btn)
        run_form.addRow("Progress:", self.run_progress)
        run_form.addRow(self.run_status_label)
        run_box.setLayout(run_form)

        self._update_mitosis_tile_size_visibility()
        self._update_havoc_tile_size_visibility()
        self._set_run_controls_enabled(True)

        selected_tile_title = QLabel("Selected Tile")
        selected_tile_title.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)

        self.viewer_folder_label = QLabel("")
        viewer_folder_font = self.viewer_folder_label.font()
        viewer_folder_font.setPointSize(11)
        self.viewer_folder_label.setFont(viewer_folder_font)
        self.viewer_folder_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)

        self.viewer_zoom_hint_label = QLabel(
            "Hover over the selected tile or WSI canvas and scroll to zoom in or out."
        )
        viewer_hint_font = self.viewer_zoom_hint_label.font()
        viewer_hint_font.setPointSize(9)
        self.viewer_zoom_hint_label.setFont(viewer_hint_font)
        self.viewer_zoom_hint_label.setStyleSheet("color: palette(mid);")
        self.viewer_zoom_hint_label.setWordWrap(True)
        self.viewer_zoom_hint_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        viewer_panel = QWidget()
        viewer_layout = QVBoxLayout()
        viewer_layout.setContentsMargins(16, 16, 16, 16)
        viewer_layout.setSpacing(14)
        viewer_layout.addWidget(selected_tile_title)
        viewer_layout.addWidget(self.tile_preview_label, 0, Qt.AlignmentFlag.AlignHCenter)
        viewer_layout.addWidget(controls)
        viewer_layout.addStretch(1)
        viewer_layout.addWidget(self.viewer_folder_label, 0, Qt.AlignmentFlag.AlignLeft)
        viewer_layout.addWidget(self.viewer_zoom_hint_label, 0, Qt.AlignmentFlag.AlignLeft)
        viewer_panel.setLayout(viewer_layout)

        run_panel = QWidget()
        run_panel_layout = QVBoxLayout()
        run_panel_layout.setContentsMargins(16, 16, 16, 16)
        run_panel_layout.setSpacing(12)
        run_panel_layout.addWidget(run_box)
        run_panel_layout.addStretch(1)
        run_panel.setLayout(run_panel_layout)

        self.mode_tabs = QTabWidget()
        self.mode_tabs.addTab(run_panel, "Run")
        self.mode_tabs.addTab(viewer_panel, "Viewer")

        root = QHBoxLayout()
        root.addWidget(self.mode_tabs, 0)
        root.addWidget(self.view, 1)

        w = QWidget()
        w.setLayout(root)
        self.setCentralWidget(w)

        self.resize(1400, 850)

    def _hbox(self, *widgets: QWidget) -> QWidget:
        c = QWidget()
        lay = QHBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        lay.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        for ww in widgets:
            lay.addWidget(ww)
        c.setLayout(lay)
        return c

    def _update_havoc_tile_size_visibility(self):
        visible = self.havoc_checkbox.isChecked()
        self.havoc_tile_size_label.setVisible(visible)
        self.havoc_tile_size_spin.setVisible(visible)

    def _update_mitosis_tile_size_visibility(self):
        visible = self.mitosis_checkbox.isChecked()
        self.mitosis_tile_size_label.setVisible(visible)
        self.mitosis_tile_size_spin.setVisible(visible)

    # ---------- Viewer ----------
    def open_heatmap_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select heatmap folder")
        if not folder:
            return

        thumb_path = os.path.join(folder, "thumbnail.jpg")
        mitosis_csv_path = os.path.join(folder, "info_df_mitosis.csv")
        havoc_csv_path = os.path.join(folder, "info_df_havoc.csv")

        if not os.path.exists(thumb_path):
            QMessageBox.warning(self, "Missing files", f"Expected:\n- {thumb_path}")
            return

        mitosis_exists = os.path.exists(mitosis_csv_path)
        havoc_exists = os.path.exists(havoc_csv_path)
        if not mitosis_exists and not havoc_exists:
            QMessageBox.warning(
                self,
                "Missing files",
                f"Expected at least one of:\n- {mitosis_csv_path}\n- {havoc_csv_path}",
            )
            return

        try:
            thumb = cv2.imread(thumb_path, cv2.IMREAD_COLOR)
            if thumb is None:
                raise RuntimeError("cv2.imread failed (thumbnail).")

            self.viewer_folder_label.setText(pathlib.Path(folder).stem)
            self.thumb_bgr = thumb
            dimmed_thumb = cv2.convertScaleAbs(thumb, alpha=0.7, beta=0)
            self.thumb_pixmap = bgr_to_qpixmap(dimmed_thumb)
            self.thumb_item.setPixmap(self.thumb_pixmap)
            self.scene.setSceneRect(0, 0, self.thumb_pixmap.width(), self.thumb_pixmap.height())
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.view.set_base_zoom()

            mitosis_df = None
            if os.path.exists(mitosis_csv_path):
                mitosis_df = pd.read_csv(mitosis_csv_path)

            if mitosis_df is not None:
                missing = [c for c in COORD_COLUMNS + ["mitosis_raw"] if c not in mitosis_df.columns]
                if missing:
                    raise ValueError(f"Mitosis CSV missing columns: {missing}")
                self.mitosis_rows = self._preprocess_rows(mitosis_df)
            else:
                self.mitosis_rows = []

            if os.path.exists(havoc_csv_path):
                havoc_df = pd.read_csv(havoc_csv_path)
                missing = [c for c in COORD_COLUMNS if c not in havoc_df.columns]
                if missing:
                    raise ValueError(f"HAVOC CSV missing columns: {missing}")
                self.havoc_df = havoc_df
            else:
                self.havoc_df = None

            self.saved_tiles = self._load_saved_tiles(os.path.join(folder, "tiles"))
            self.saved_tile_map = {(tile.coords[0], tile.coords[1]): tile for tile in self.saved_tiles}
            self.current_saved_tile = self.saved_tiles[0] if self.saved_tiles else None
            self.viewer_tile_size = (
                self.saved_tiles[0].coords[2] - self.saved_tiles[0].coords[0]
                if self.saved_tiles
                else 512
            )

            self._reset_viewer_overlays()
            self._update_selection_rect(None)
            self.havoc_box.setVisible(self.havoc_df is not None)
            self.mitosis_box.setVisible(bool(self.mitosis_rows))
            self.havoc_box.setChecked(self.havoc_df is not None)
            self.mitosis_box.setChecked(bool(self.mitosis_rows))

            if self.current_saved_tile is not None:
                self._show_saved_tile_preview(self.current_saved_tile)
                self._update_selection_rect(self.current_saved_tile)
            else:
                self._clear_saved_tile_preview()

            if self.havoc_df is not None:
                self._refresh_havoc_overlay()
            if self.mitosis_rows:
                self.request_mitosis_render()

            self.mode_tabs.setCurrentIndex(1)
            self.view.setFocus()

        except Exception as e:
            QMessageBox.critical(self, "Load error", f"{type(e).__name__}: {e}")

    def _load_saved_tiles(self, tiles_dir: str) -> List[SavedTile]:
        if not os.path.isdir(tiles_dir):
            return []

        out: List[SavedTile] = []
        for name in os.listdir(tiles_dir):
            tile_path = os.path.join(tiles_dir, name)
            if not os.path.isfile(tile_path):
                continue
            if pathlib.Path(tile_path).suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
                continue
            out.append(SavedTile(path=tile_path, coords=parse_tile_coords(tile_path)))

        out.sort(key=lambda tile: tile.coords)
        return out

    def _preprocess_rows(self, df: pd.DataFrame) -> List[RowDatum]:
        out: List[RowDatum] = []
        for _, row in df.iterrows():
            x1 = int(row["coor_x1_20x"])
            y1 = int(row["coor_y1_20x"])
            x2 = int(row["coor_x2_20x"])
            y2 = int(row["coor_y2_20x"])

            vals = parse_scores_str(str(row["mitosis_raw"]), lower=0.4, upper=1.0)
            if vals.size:
                vals_sorted = np.sort(vals)
                suffix = np.cumsum(vals_sorted[::-1], dtype=np.float32)[::-1]
            else:
                vals_sorted = np.array([], dtype=np.float32)
                suffix = np.array([], dtype=np.float32)

            out.append(RowDatum(x1, y1, x2, y2, vals_sorted, suffix))
        return out

    def _thr_slider_changed(self, v: int):
        self.thr_spin.blockSignals(True)
        self.thr_spin.setValue(v / 100.0)
        self.thr_spin.blockSignals(False)
        self.request_mitosis_render()

    def _thr_spin_changed(self, v: float):
        self.thr_slider.blockSignals(True)
        self.thr_slider.setValue(int(round(v * 100)))
        self.thr_slider.blockSignals(False)
        self.request_mitosis_render()

    def _mitosis_opacity_changed(self, v: int):
        self.mitosis_overlay_item.setOpacity(v / 100.0)

    def _havoc_opacity_changed(self, v: int):
        self.havoc_overlay_item.setOpacity(v / 100.0)

    def _havoc_k_changed(self, v: int):
        self.havoc_k_label.setText(f"k = {v}")
        if self.havoc_box.isVisible() and self.havoc_box.isChecked():
            self._refresh_havoc_overlay()

    def request_mitosis_render(self):
        if self.thumb_bgr is None or not self.mitosis_rows or not self.mitosis_box.isChecked():
            self.mitosis_overlay_item.setPixmap(QPixmap())
            return
        self.render_timer.start(120)

    def _kick_mitosis_render(self):
        if self.thumb_bgr is None or not self.mitosis_rows or not self.mitosis_box.isChecked():
            return

        thr = self.thr_slider.value() / 100.0
        # sigma = float(self.blur_sigma_spin.value()) if self.blur_check.isChecked() else 0.0
        sigma = 0.0
        key = (round(thr, 2), round(sigma, 1))

        if key in self.mitosis_overlay_cache:
            self.mitosis_overlay_item.setPixmap(self.mitosis_overlay_cache[key])
            self.mitosis_overlay_item.setOpacity(self.mitosis_opacity_slider.value() / 100.0)
            return

        task = RenderTask(
            thumb_bgr=self.thumb_bgr,
            rows=self.mitosis_rows,
            scale_factor=self.scale_factor,
            score_threshold=key[0],
            blur_sigma=key[1],
        )
        task.signals.rendered.connect(self._render_done)
        task.signals.failed.connect(self._render_failed)
        self.thread_pool.start(task)

    def _render_done(self, thr: float, sigma: float, pm: QPixmap):
        key = (round(float(thr), 2), round(float(sigma), 1))
        self._cache_put_mitosis(key, pm)
        if self.mitosis_box.isChecked():
            self.mitosis_overlay_item.setPixmap(pm)
            self.mitosis_overlay_item.setOpacity(self.mitosis_opacity_slider.value() / 100.0)

    def _render_failed(self, msg: str):
        QMessageBox.critical(self, "Render failed", msg)

    def _reset_viewer_overlays(self):
        self.mitosis_overlay_cache.clear()
        self.mitosis_cache_order.clear()
        self.havoc_overlay_cache.clear()
        self.mitosis_overlay_item.setPixmap(QPixmap())
        self.havoc_overlay_item.setPixmap(QPixmap())

    def _cache_put_mitosis(self, key: Tuple[float, float], pm: QPixmap):
        if key in self.mitosis_overlay_cache:
            return
        self.mitosis_overlay_cache[key] = pm
        self.mitosis_cache_order.append(key)
        while len(self.mitosis_cache_order) > self.mitosis_cache_max_items:
            old = self.mitosis_cache_order.pop(0)
            self.mitosis_overlay_cache.pop(old, None)

    def _refresh_havoc_overlay(self):
        if self.thumb_bgr is None or self.havoc_df is None or not self.havoc_box.isChecked():
            self.havoc_overlay_item.setPixmap(QPixmap())
            return

        k_val = int(self.havoc_k_slider.value())
        if k_val in self.havoc_overlay_cache:
            pm = self.havoc_overlay_cache[k_val]
        else:
            pm = build_havoc_overlay_pixmap(self.thumb_pixmap, self.havoc_df, self.scale_factor, k_val)
            self.havoc_overlay_cache[k_val] = pm

        self.havoc_overlay_item.setPixmap(pm)
        self.havoc_overlay_item.setOpacity(self.havoc_opacity_slider.value() / 100.0)

    def _update_havoc_overlay_visibility(self, checked: bool):
        if checked:
            self._refresh_havoc_overlay()
        else:
            self.havoc_overlay_item.setPixmap(QPixmap())

    def _update_mitosis_overlay_visibility(self, checked: bool):
        if checked:
            self.request_mitosis_render()
        else:
            self.mitosis_overlay_item.setPixmap(QPixmap())

    def _viewer_clicked(self, x: float, y: float):
        if not self.saved_tiles:
            return

        for saved_tile in self.saved_tiles:
            disp_x1 = saved_tile.coords[0] / self.scale_factor
            disp_y1 = saved_tile.coords[1] / self.scale_factor
            disp_x2 = saved_tile.coords[2] / self.scale_factor
            disp_y2 = saved_tile.coords[3] / self.scale_factor
            if disp_x1 <= x <= disp_x2 and disp_y1 <= y <= disp_y2:
                self.current_saved_tile = saved_tile
                self._show_saved_tile_preview(saved_tile)
                self._update_selection_rect(saved_tile)
                return

    def _viewer_key_pressed(self, key: int):
        if not self.saved_tiles:
            return

        if self.current_saved_tile is None:
            self.current_saved_tile = self.saved_tiles[0]

        curr_x = self.current_saved_tile.coords[0]
        curr_y = self.current_saved_tile.coords[1]
        if key == Qt.Key.Key_Left:
            next_key = (curr_x - self.viewer_tile_size, curr_y)
        elif key == Qt.Key.Key_Right:
            next_key = (curr_x + self.viewer_tile_size, curr_y)
        elif key == Qt.Key.Key_Up:
            next_key = (curr_x, curr_y - self.viewer_tile_size)
        elif key == Qt.Key.Key_Down:
            next_key = (curr_x, curr_y + self.viewer_tile_size)
        else:
            return

        next_tile = self.saved_tile_map.get(next_key)
        if next_tile is None:
            return

        self.current_saved_tile = next_tile
        self._show_saved_tile_preview(next_tile)
        self._update_selection_rect(next_tile)

    def _update_selection_rect(self, saved_tile: Optional[SavedTile]):
        if self.selection_rect_item is not None and self.selection_rect_item.scene() == self.scene:
            self.scene.removeItem(self.selection_rect_item)
        self.selection_rect_item = None

        if saved_tile is None:
            return

        disp_x1 = saved_tile.coords[0] / self.scale_factor
        disp_y1 = saved_tile.coords[1] / self.scale_factor
        disp_w = (saved_tile.coords[2] - saved_tile.coords[0]) / self.scale_factor
        disp_h = (saved_tile.coords[3] - saved_tile.coords[1]) / self.scale_factor

        self.selection_rect_item = QGraphicsRectItem(disp_x1, disp_y1, disp_w, disp_h)
        pen = QPen(QColor(255, 255, 255))
        pen.setWidth(2)
        self.selection_rect_item.setPen(pen)
        self.scene.addItem(self.selection_rect_item)

    def _show_saved_tile_preview(self, saved_tile: Optional[SavedTile]):
        if saved_tile is None or not os.path.exists(saved_tile.path):
            self._clear_saved_tile_preview()
            return

        pm = QPixmap(saved_tile.path)
        if pm.isNull():
            self._clear_saved_tile_preview()
            return

        self.tile_preview_label.setText("")
        self.tile_preview_label.setPixmap(pm)
        self.tile_preview_label.show()

    def _clear_saved_tile_preview(self):
        self.tile_preview_label.clear()
        self.tile_preview_label.setText("Selected Tile")
        self.tile_preview_label.show()

    # ---------- Run ----------
    def browse_slides_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select slides folder")
        if not folder:
            return
        self.slides_folder_edit.setText(folder)
        default_out = os.path.join(folder, "_out")
        self.output_root_edit.setText(default_out)
        self.run_status_label.setText("Ready to run. (Will generate per-slide subfolders in the output root.)")
        self.run_progress.setValue(0)

    def _set_run_controls_enabled(self, enabled: bool, show_cancel: bool = False, cancel_enabled: bool = False):
        self.slides_folder_edit.setEnabled(enabled)
        self.output_root_edit.setEnabled(enabled)
        self.browse_slides_btn.setEnabled(enabled)
        self.mitosis_checkbox.setEnabled(enabled)
        self.mitosis_tile_size_label.setEnabled(enabled)
        self.mitosis_tile_size_spin.setEnabled(False)
        self.havoc_checkbox.setEnabled(enabled)
        self.havoc_tile_size_label.setEnabled(enabled)
        self.havoc_tile_size_spin.setEnabled(enabled and self.havoc_checkbox.isChecked())
        self.run_cancel_mode = bool(show_cancel)
        self.run_btn.setText("Cancel" if show_cancel else "Run slide processing")
        self.run_btn.setEnabled(enabled or cancel_enabled)

    def _handle_run_button_clicked(self):
        if self.run_cancel_mode:
            self._cancel_active_process()
            return
        self.run_processing()

    def _cancel_active_process(self):
        if self.run_process is None:
            return
        self.run_cancel_requested = True
        self.run_btn.setEnabled(False)
        self.run_status_label.setText("Cancelling run...")
        self.run_process.terminate()
        QTimer.singleShot(3000, self._force_kill_run_process_if_needed)

    def _force_kill_run_process_if_needed(self):
        if self.run_process is not None and self.run_process.state() != QProcess.ProcessState.NotRunning:
            self.run_process.kill()

    def _cancel_pending_run(self, status: str):
        self.pending_run_config = None
        self.run_cancel_requested = False
        self._set_run_controls_enabled(True)
        self.run_progress.setRange(0, 100)
        self.run_progress.setValue(0)
        self.run_status_label.setText(status)

    def _start_slide_run(self, config: PendingRunConfig):
        self._set_run_controls_enabled(False, show_cancel=True, cancel_enabled=False)
        self.run_progress.setRange(0, 100)
        self.run_progress.setValue(0)
        self.run_status_label.setText("Starting...")
        self.mode_tabs.setCurrentIndex(0)

        fd, config_path = tempfile.mkstemp(prefix="onsight_run_", suffix=".json")
        os.close(fd)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "slide_paths": config.slide_paths,
                    "mpp_overrides": config.mpp_overrides,
                    "output_root": config.output_root,
                    "run_mitosis": config.run_mitosis,
                    "run_havoc": config.run_havoc,
                    "havoc_tile_size": config.havoc_tile_size,
                },
                f,
            )

        self.run_process_config_path = config_path
        self.run_cancel_requested = False
        self.run_stdout_buffer = ""
        self.run_stderr_buffer = ""
        self.run_completed_out_root = None
        self.run_failed_message = None

        proc = QProcess(self)
        proc.setWorkingDirectory(os.path.dirname(os.path.abspath(__file__)))
        proc.readyReadStandardOutput.connect(self._on_run_process_stdout)
        proc.readyReadStandardError.connect(self._on_run_process_stderr)
        proc.finished.connect(self._on_run_process_finished)
        proc.errorOccurred.connect(self._on_run_process_error)

        if getattr(sys, "frozen", False):
            program = sys.executable
            arguments = ["--run-job", config_path]
        else:
            program = sys.executable
            arguments = [os.path.abspath(__file__), "--run-job", config_path]

        self.run_process = proc
        proc.start(program, arguments)
        self._set_run_controls_enabled(False, show_cancel=True, cancel_enabled=True)

    def _prompt_hf_token(self) -> Optional[str]:
        dlg = HuggingFaceTokenDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        return dlg.verified_token

    def _begin_havoc_download(self, token: str):
        self._set_run_controls_enabled(False, show_cancel=True, cancel_enabled=False)
        self.run_progress.setRange(0, 0)
        self.run_status_label.setText("Downloading Prov-GigaPath model...")

        self.hf_download_worker = HuggingFaceDownloadWorker(token, self)
        self.hf_download_worker.status.connect(self._hf_download_status)
        self.hf_download_worker.download_finished.connect(self._hf_download_finished)
        self.hf_download_worker.error.connect(self._hf_download_failed)

        self.hf_download_dialog = HuggingFaceDownloadDialog(self.hf_download_worker, self)
        self.hf_download_dialog.show()
        self.hf_download_worker.start()

    def _hf_download_status(self, msg: str):
        if self.sender() is not self.hf_download_worker:
            return
        self.run_status_label.setText(msg)

    def _hf_download_finished(self):
        if self.sender() is not self.hf_download_worker:
            return
        self.run_progress.setRange(0, 100)
        self.run_progress.setValue(100)
        self.run_status_label.setText("HAVOC model ready. Starting slide processing...")
        config = self.pending_run_config
        self.pending_run_config = None
        self.hf_download_worker = None
        self.hf_download_dialog = None
        if config is not None:
            self._start_slide_run(config)

    def _hf_download_failed(self, msg: str):
        if self.sender() is not self.hf_download_worker:
            return
        self.hf_download_worker = None
        self.hf_download_dialog = None
        self._cancel_pending_run("HAVOC download failed.")
        QMessageBox.critical(self, "Download Failed", msg)

    def _prepare_havoc_requirements(self):
        try:
            token = get_hf_token()
        except Exception as e:
            self._cancel_pending_run("HAVOC setup failed.")
            QMessageBox.critical(self, "HAVOC setup failed", f"{type(e).__name__}: {e}")
            return

        if token:
            try:
                login_huggingface(token)
            except Exception:
                token = self._prompt_hf_token()
        else:
            token = self._prompt_hf_token()

        if not token:
            self._cancel_pending_run("Run cancelled.")
            return

        self.run_status_label.setText("Checking Prov-GigaPath cache...")
        QApplication.processEvents()

        try:
            if is_havoc_model_downloaded(token):
                config = self.pending_run_config
                self.pending_run_config = None
                if config is not None:
                    self._start_slide_run(config)
                return
        except Exception as e:
            self._cancel_pending_run("HAVOC setup failed.")
            QMessageBox.critical(self, "HAVOC setup failed", f"{type(e).__name__}: {e}")
            return

        self._begin_havoc_download(token)

    def run_processing(self):
        slides_folder = self.slides_folder_edit.text().strip()
        out_root = self.output_root_edit.text().strip()
        run_mitosis = self.mitosis_checkbox.isChecked()
        run_havoc = self.havoc_checkbox.isChecked()

        if not slides_folder or not os.path.isdir(slides_folder):
            QMessageBox.warning(self, "Missing input", "Select a valid slides folder first.")
            return
        if not out_root:
            QMessageBox.warning(self, "Missing output", "Select an output root folder first.")
            return
        if not run_mitosis and not run_havoc:
            QMessageBox.warning(self, "Missing pipeline", "Select at least one of Mitosis or HAVOC.")
            return

        havoc_tile_size = None
        if run_havoc:
            havoc_tile_size = int(self.havoc_tile_size_spin.value())

        slide_paths = self._find_slides(slides_folder)
        if not slide_paths:
            QMessageBox.warning(self, "No slides found", f"No supported slide files found in:\n{slides_folder}")
            return
        self._set_run_controls_enabled(False, show_cancel=True, cancel_enabled=False)

        # Pre-scan for missing mpp (and validate slides can be opened)
        missing_mpp: List[str] = []
        self.run_status_label.setText("Scanning slides for MPP metadata…")
        QApplication.processEvents()

        for sp in slide_paths:
            try:
                s = Slide(sp)
                if s.mpp is None:
                    missing_mpp.append(sp)
            except Exception as e:
                self._cancel_pending_run("Run failed.")
                QMessageBox.critical(self, "Slide open failed", f"{os.path.basename(sp)}\n{type(e).__name__}: {e}")
                return

        # Collect per-slide overrides for missing slides
        mpp_overrides: Dict[str, float] = {}
        if missing_mpp:
            dlg = MppOverrideDialog(missing_mpp, parent=self)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                self._cancel_pending_run("Run cancelled.")
                return
            mpp_overrides = dlg.get_overrides()

        config = PendingRunConfig(
            slide_paths=slide_paths,
            mpp_overrides=mpp_overrides,
            output_root=out_root,
            run_mitosis=run_mitosis,
            run_havoc=run_havoc,
            havoc_tile_size=havoc_tile_size,
        )

        if run_havoc:
            self.pending_run_config = config
            self._set_run_controls_enabled(False)
            self.run_progress.setRange(0, 0)
            self.run_status_label.setText("Preparing HAVOC model...")
            self.mode_tabs.setCurrentIndex(0)
            self._prepare_havoc_requirements()
            return

        self._start_slide_run(config)

    def _find_slides(self, folder: str) -> List[str]:
        out = []
        for name in sorted(os.listdir(folder)):
            p = os.path.join(folder, name)
            if not os.path.isfile(p):
                continue
            ext = pathlib.Path(p).suffix.lower()
            if ext in SUPPORTED_SLIDE_EXTS:
                out.append(p)
        return out

    def _run_started(self, n: int):
        self.run_progress.setRange(0, n)
        self.run_progress.setValue(0)
        self.run_status_label.setText(f"Running on {n} slide(s)…")

    def _cleanup_run_process(self):
        if self.run_process is not None:
            try:
                self.run_process.deleteLater()
            except Exception:
                pass
        self.run_process = None

        if self.run_process_config_path and os.path.exists(self.run_process_config_path):
            try:
                os.remove(self.run_process_config_path)
            except OSError:
                pass
        self.run_process_config_path = None

    def _handle_run_event(self, event: dict):
        event_type = event.get("type")
        if event_type == "started":
            self._run_started(int(event.get("n", 0)))
        elif event_type == "progress":
            self._run_progress_msg(str(event.get("msg", "")))
        elif event_type == "slide_done":
            self._run_slide_done(str(event.get("slide_name", "")), str(event.get("out_dir", "")))
        elif event_type == "finished":
            self.run_completed_out_root = str(event.get("out_root", ""))
        elif event_type == "failed":
            self.run_failed_message = str(event.get("msg", "Run failed."))

    def _consume_run_process_text(self, text: str):
        self.run_stdout_buffer += text
        lines = self.run_stdout_buffer.splitlines(keepends=True)
        remainder = ""
        if lines and not (lines[-1].endswith("\n") or lines[-1].endswith("\r")):
            remainder = lines.pop()
        self.run_stdout_buffer = remainder

        for raw_line in lines:
            line = raw_line.strip()
            if not line.startswith(RUN_EVENT_PREFIX):
                continue
            payload = line[len(RUN_EVENT_PREFIX):]
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            self._handle_run_event(event)

    def _on_run_process_stdout(self):
        if self.run_process is None:
            return
        text = bytes(self.run_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._consume_run_process_text(text)

    def _on_run_process_stderr(self):
        if self.run_process is None:
            return
        text = bytes(self.run_process.readAllStandardError()).decode("utf-8", errors="replace")
        self.run_stderr_buffer += text

    def _on_run_process_error(self, _error):
        if self.run_process is None:
            return
        if self.run_cancel_requested:
            return
        self.run_failed_message = self.run_failed_message or "Failed to start slide processing subprocess."

    def _on_run_process_finished(self, exit_code: int, exit_status):
        if self.run_stdout_buffer:
            self._consume_run_process_text("\n")

        completed_out_root = self.run_completed_out_root
        failed_message = self.run_failed_message
        stderr_text = self.run_stderr_buffer.strip()

        self._cleanup_run_process()
        self.run_stdout_buffer = ""
        self.run_stderr_buffer = ""
        self.run_completed_out_root = None
        self.run_failed_message = None

        if self.run_cancel_requested:
            self.run_cancel_requested = False
            self._cancel_pending_run("Run cancelled.")
            return

        if exit_status == QProcess.ExitStatus.NormalExit and exit_code == 0 and completed_out_root:
            self._run_finished(completed_out_root)
            return

        if not failed_message:
            failed_message = stderr_text or f"Slide-processing subprocess exited with code {exit_code}."
        self._run_failed(failed_message)

    def _run_progress_msg(self, msg: str):
        self.run_status_label.setText(msg)

    def _run_slide_done(self, slide_name: str, out_dir: str):
        self.run_progress.setValue(self.run_progress.value() + 1)

    def _run_finished(self, out_root: str):
        self._set_run_controls_enabled(True)
        self.run_status_label.setText("Done.")
        QMessageBox.information(self, "Run complete", f"Generated outputs under:\n{out_root}")

    def _run_failed(self, msg: str):
        self._set_run_controls_enabled(True)
        self.run_status_label.setText("Run failed.")
        QMessageBox.critical(self, "Run failed", msg)


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--run-job":
        try:
            return run_job_from_config(sys.argv[2])
        except Exception as e:
            emit_run_event("failed", msg=f"{type(e).__name__}: {e}")
            traceback.print_exc()
            return 1

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
