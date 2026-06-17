# Custom Enlarge Image Dialog for OnSight Pathology

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QDialog, QScrollArea, QSplitter, QSizePolicy
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor
from PyQt6.QtCore import Qt, QTimer, QSize


# ----------------------------------------------------------------------
# Paint-driven image canvas. Replaces the old QLabel-with-setPixmap
# approach to eliminate flicker, mis-coloured letterbox bars during
# resize, and the stall caused by re-doing SmoothTransformation on
# every single resize event.
# ----------------------------------------------------------------------
class _ImageCanvas(QWidget):
    BACKGROUND     = QColor("#111418")           # near-black; letterbox bars look intentional
    OVERLAY_BG     = QColor(255, 255, 255, 240)  # almost-opaque white card behind the thumbnail
    OVERLAY_BORDER = QColor("#dcdde1")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = QPixmap()
        self._overlay = QPixmap()

        # Cache of the most recently scaled pixmap so we don't redo
        # SmoothTransformation on every paint when nothing changed.
        self._cached_scaled = QPixmap()
        self._cached_for_size = QSize()
        self._cached_smooth = True

        # Resize state — paint at FastTransformation while the user
        # is actively dragging the window edge, then re-paint at
        # SmoothTransformation ~90ms after they stop moving.
        self._is_resizing = False
        self._settle_timer = QTimer(self)
        self._settle_timer.setSingleShot(True)
        self._settle_timer.setInterval(90)
        self._settle_timer.timeout.connect(self._on_resize_settled)

        # Skip Qt's automatic background clear — we paint every pixel
        # ourselves, so the auto-clear would just be a wasted white flash.
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(160, 120)

    # ------------------------------------------------------------------
    # Content setters
    # ------------------------------------------------------------------
    def set_pixmap(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self._cached_for_size = QSize()  # invalidate cache — pixmap changed
        self.update()

    def set_overlay(self, pixmap: QPixmap):
        self._overlay = pixmap
        self.update()

    def clear_overlay(self):
        self._overlay = QPixmap()
        self.update()

    # ------------------------------------------------------------------
    # Resize handling — FastTransformation during, Smooth after.
    # ------------------------------------------------------------------
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._is_resizing = True
        self._settle_timer.start()
        self.update()

    def _on_resize_settled(self):
        self._is_resizing = False
        self._cached_for_size = QSize()  # force one final SmoothTransformation
        self.update()

    # ------------------------------------------------------------------
    # Single paint pass: background → fitted pixmap → overlay thumbnail.
    # Doing everything here means no transient state with one layer
    # painted and another not yet — i.e., no flashes of background
    # colour during resize or frame updates.
    # ------------------------------------------------------------------
    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.BACKGROUND)

        if not self._pixmap.isNull():
            want_smooth = not self._is_resizing
            if (self._cached_for_size != self.size()
                    or self._cached_smooth != want_smooth
                    or self._cached_scaled.isNull()):
                mode = (Qt.TransformationMode.SmoothTransformation
                        if want_smooth else
                        Qt.TransformationMode.FastTransformation)
                self._cached_scaled = self._pixmap.scaled(
                    self.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    mode,
                )
                self._cached_for_size = self.size()
                self._cached_smooth = want_smooth

            pix = self._cached_scaled
            x = (self.width() - pix.width()) // 2
            y = (self.height() - pix.height()) // 2
            painter.drawPixmap(x, y, pix)

        # Overlay thumbnail in the top-right corner of the canvas.
        if not self._overlay.isNull():
            side = max(100, min(self.width(), self.height()) // 3)
            scaled_ov = self._overlay.scaled(
                side, side,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            margin = 10
            pad = 4
            ox = self.width() - scaled_ov.width() - margin
            oy = margin
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(self.OVERLAY_BORDER)
            painter.setBrush(self.OVERLAY_BG)
            painter.drawRoundedRect(
                ox - pad, oy - pad,
                scaled_ov.width() + 2 * pad,
                scaled_ov.height() + 2 * pad,
                8, 8,
            )
            painter.drawPixmap(ox, oy, scaled_ov)


##############################################################################################################################################
# Custom Enlarge Image Dialog for OnSight Pathology
##############################################################################################################################################
class ResizableImageDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enlarged Inference View (Live)")

        # Allow shrinking far below the previous 700x800 floor — the
        # canvas and splitter both handle small sizes gracefully.
        self.setMinimumSize(420, 340)
        self.resize(720, 820)  # pleasant default; user can drag any direction

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(8)

        # Splitter replaces the old fixed 4:1 stretch ratio. The user
        # can drag the handle to dedicate as much vertical space to the
        # image or the text as they like — handles the "I want more
        # image and less text" case the old layout couldn't.
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(6)
        self.splitter.setStyleSheet("""
            QSplitter::handle:vertical {
                background-color: #d9dee3;
                margin: 2px 4px;
                border-radius: 2px;
            }
            QSplitter::handle:vertical:hover {
                background-color: #adb5bd;
            }
        """)

        # --- top: image canvas ---
        self.image_widget = _ImageCanvas()
        self.splitter.addWidget(self.image_widget)

        # --- bottom: text panel ---
        self.text_scroll = QScrollArea()
        self.text_scroll.setWidgetResizable(True)
        self.text_scroll.setMinimumHeight(60)
        self.text_scroll.setStyleSheet("border: none; background-color: transparent;")

        self.text_label = QLabel("Waiting for data...")
        self.text_label.setTextFormat(Qt.TextFormat.RichText)
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.text_label.setStyleSheet(
            "background-color: white; color: black; padding: 10px; border-radius: 4px;"
        )
        self.text_label.setWordWrap(True)
        self.text_scroll.setWidget(self.text_label)

        self.splitter.addWidget(self.text_scroll)
        # Initial proportion roughly matches the previous 4:1 feel,
        # but the user can drag the handle to whatever they prefer.
        self.splitter.setStretchFactor(0, 4)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([640, 160])

        self.main_layout.addWidget(self.splitter)

        # Kept as public attributes in case any external code touches
        # them — same names as before, no behavioural dependency.
        self.original_pixmap = QPixmap()
        self.latest_overlay_rgb = None

    # ------------------------------------------------------------------
    # Public API — signatures unchanged from the original dialog.
    # ------------------------------------------------------------------
    def set_image(self, image: np.ndarray):
        if image is None or image.size == 0:
            return
        h, w, c = image.shape
        # Ensure contiguous so QImage's stride math is correct.
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)
        qimg = QImage(image.data, w, h, c * w, QImage.Format.Format_RGB888)
        # .copy() detaches from the numpy buffer so the pixmap survives
        # the caller overwriting `image` on the next frame.
        self.original_pixmap = QPixmap.fromImage(qimg.copy())
        self.image_widget.set_pixmap(self.original_pixmap)

    def set_text(self, text: str):
        self.text_label.setText(text)

    def set_overlay(self, result_rgb: np.ndarray):
        """Receive an overlay result from the worker thread, store it,
        and draw it as a thumbnail in the top-right of the image area."""
        if result_rgb is None or result_rgb.size == 0:
            return
        self.latest_overlay_rgb = result_rgb.copy()
        h, w, c = result_rgb.shape
        rgb_data = np.ascontiguousarray(result_rgb)
        qimg = QImage(rgb_data.data, w, h, c * w, QImage.Format.Format_RGB888)
        self.image_widget.set_overlay(QPixmap.fromImage(qimg.copy()))

    def hide_overlay(self):
        """Called from the main window when Stop is pressed."""
        self.latest_overlay_rgb = None
        self.image_widget.clear_overlay()