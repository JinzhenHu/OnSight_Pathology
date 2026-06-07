"""
Simple loading dialog for model downloads.
Clean, light theme, no animations or fancy effects.
"""
import time
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QProgressBar
)
from PyQt6.QtCore import Qt, pyqtSignal


LOADING_STYLESHEET = """
QDialog {
    background-color: #ffffff;
}

QLabel#TitleLabel {
    color: #1a1a1a;
    font-size: 13pt;
    font-weight: bold;
}

QLabel#ModelLabel {
    color: #2c3e50;
    font-size: 10pt;
    font-weight: bold;
}

QLabel#StatusLabel {
    color: #555555;
    font-size: 9pt;
}

QLabel#StatsLabel {
    color: #888888;
    font-size: 8pt;
    font-family: 'Consolas', 'Menlo', monospace;
}

QProgressBar {
    background-color: #f0f0f0;
    border: 1px solid #cccccc;
    border-radius: 4px;
    height: 18px;
    text-align: center;
    color: #1a1a1a;
}

QProgressBar::chunk {
    background-color: #3498db;
    border-radius: 3px;
}

QPushButton {
    background-color: #f0f0f0;
    color: #1a1a1a;
    border: 1px solid #cccccc;
    padding: 6px 18px;
    border-radius: 4px;
    min-width: 80px;
}

QPushButton:hover {
    background-color: #e0e0e0;
}

QPushButton:pressed {
    background-color: #d0d0d0;
}

QPushButton:disabled {
    color: #999999;
    background-color: #f8f8f8;
}
"""


def _fmt_bytes(n: float) -> str:
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / 1024 / 1024:.1f} MB"
    return f"{n / 1024 / 1024 / 1024:.2f} GB"


def _fmt_speed(bps: float) -> str:
    if bps <= 0:
        return "—"
    if bps < 1024 * 1024:
        return f"{bps / 1024:.0f} KB/s"
    return f"{bps / 1024 / 1024:.1f} MB/s"


def _fmt_eta(seconds: float) -> str:
    if seconds <= 0 or seconds == float("inf"):
        return "—"
    if seconds < 60:
        return f"~{int(seconds)}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"~{m}m {s}s"
    h, rem = divmod(int(seconds), 3600)
    return f"~{h}h {rem // 60}m"


class LoadingDialog(QDialog):
    """
    Simple progress dialog for model loading/downloading.
    """
    cancelled = pyqtSignal()

    def __init__(self, title: str = "Loading Model",
                 model_name: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(440)
        # Standard window with title bar; remove the help "?" button
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint
        )
        self.setStyleSheet(LOADING_STYLESHEET)

        # Stats tracking
        self._last_bytes = 0
        self._last_time = time.time()
        self._speed_bps = 0.0
        self._indeterminate = True
        self._cancel_requested = False

        self._build_ui(title, model_name)

    def _build_ui(self, title: str, model_name: str):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(10)

        self.lbl_title = QLabel(title)
        self.lbl_title.setObjectName("TitleLabel")
        layout.addWidget(self.lbl_title)

        self.lbl_model = QLabel(model_name)
        self.lbl_model.setObjectName("ModelLabel")
        self.lbl_model.setWordWrap(True)
        layout.addWidget(self.lbl_model)

        self.lbl_status = QLabel("Preparing…")
        self.lbl_status.setObjectName("StatusLabel")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(18)
        layout.addWidget(self.bar)

        self.lbl_stats = QLabel(" ")
        self.lbl_stats.setObjectName("StatsLabel")
        self.lbl_stats.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_stats)

        layout.addSpacing(4)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self._on_cancel)
        btn_row.addWidget(self.btn_cancel)
        layout.addLayout(btn_row)

    # ---------- Public API ----------
    def set_model_name(self, name: str):
        self.lbl_model.setText(name)

    def set_status(self, text: str):
        self.lbl_status.setText(text)

    def set_progress(self, percent: int,
                     current_bytes: float = 0,
                     total_bytes: float = 0):
        """percent: 0..100 for determinate, -1 for indeterminate."""
        now = time.time()

        if percent < 0:
            if not self._indeterminate:
                self._indeterminate = True
                self.bar.setRange(0, 0)
            self.lbl_stats.setText("Initializing…")
            return

        if self._indeterminate:
            self._indeterminate = False
            self.bar.setRange(0, 100)

        self.bar.setValue(percent)

        # Update speed (sliding-window average)
        if current_bytes > 0:
            dt = now - self._last_time
            if dt >= 0.4:
                db = current_bytes - self._last_bytes
                inst = db / dt if dt > 0 else 0
                self._speed_bps = (0.7 * inst + 0.3 * self._speed_bps) if self._speed_bps else inst
                self._last_bytes = current_bytes
                self._last_time = now

        if total_bytes > 0:
            remaining = max(0, total_bytes - current_bytes)
            eta = remaining / self._speed_bps if self._speed_bps > 0 else -1
            self.lbl_stats.setText(
                f"{_fmt_bytes(current_bytes)} / {_fmt_bytes(total_bytes)}"
                f"   •   {_fmt_speed(self._speed_bps)}"
                f"   •   {_fmt_eta(eta)}"
            )
        else:
            self.lbl_stats.setText(f"{percent}%")

    def is_cancel_requested(self) -> bool:
        return self._cancel_requested

    # ---------- Cancel handling ----------
    def _on_cancel(self):
        self._cancel_requested = True
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setText("Cancelling…")
        self.lbl_status.setText("Cancellation requested, please wait…")
        self.cancelled.emit()

    # Intercept close (X button) so it behaves like cancel
    def closeEvent(self, event):
        if not self._cancel_requested:
            self._cancel_requested = True
            self.cancelled.emit()
            # Don't close immediately - wait for the loader thread to finish.
            # The caller is responsible for closing this dialog via accept()/reject().
            event.ignore()
        else:
            event.accept()