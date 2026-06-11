"""
Windows-only status bar indicator for current display scaling.
Shows nothing on macOS / Linux (they don't have this problem).
"""
import os
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame, QLabel, QHBoxLayout, QVBoxLayout, QDialog,
    QPushButton, QSizePolicy, QMessageBox, QApplication,
)

from custom_widgets.DpiWarningDialog import _current_dpi_scale


def install_dpi_indicator(main_window) -> None:
    """Attach the indicator to a QMainWindow's status bar.
    No-op on macOS and Linux. Call this from your app's __init__.
    """
    if sys.platform != "win32":
        return
    indicator = DpiStatusIndicator(main_window)
    main_window.statusBar().addPermanentWidget(indicator)
    main_window.dpi_indicator = indicator


class DpiStatusIndicator(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("dpiPill")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(22)

        h = QHBoxLayout(self)
        h.setContentsMargins(10, 0, 10, 0)
        h.setSpacing(6)

        self._dot = QLabel("●")
        self._dot.setStyleSheet("background: transparent;")
        f = QFont()
        f.setPointSize(8)
        self._dot.setFont(f)
        h.addWidget(self._dot, alignment=Qt.AlignmentFlag.AlignVCenter)

        self._lbl = QLabel()
        self._lbl.setStyleSheet("background: transparent;")
        lf = QFont()
        lf.setPointSize(9)
        self._lbl.setFont(lf)
        h.addWidget(self._lbl, alignment=Qt.AlignmentFlag.AlignVCenter)

        self._refresh()

    def _refresh(self):
        scale = _current_dpi_scale()
        pct = int(round(scale * 100))
        self._is_ok = abs(scale - 1.0) < 0.05
        self._pct = pct

        if self._is_ok:
            self._dot.setStyleSheet("color: #2e8b57; background: transparent;")
            self._lbl.setText(f"Display scaling: {pct}%")
            self._lbl.setStyleSheet("color: #2e8b57; background: transparent;")
            self.setStyleSheet(
                "#dpiPill { background: transparent; border: none; }"
                "#dpiPill:hover { background: rgba(46,139,87,0.10); border-radius: 3px; }"
            )
            self.setToolTip("Display scaling is 100%. Click for details.")
        else:
            self._dot.setStyleSheet("color: #c97a2b; background: transparent;")
            self._lbl.setText(f"Display scaling: {pct}%")
            self._lbl.setStyleSheet("color: #c97a2b; background: transparent;")
            self.setStyleSheet(
                "#dpiPill { background: transparent; border: none; }"
                "#dpiPill:hover { background: rgba(201,122,43,0.12); border-radius: 3px; }"
            )
            self.setToolTip("Display scaling is not 100%. Click for details.")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            dlg = DpiDetailDialog(self._is_ok, self._pct, self.window())
            dlg.exec()
        super().mousePressEvent(event)


# ---------------------------------------------------------------------------
class DpiDetailDialog(QDialog):
    AFFECTED = [
        ("Magnification detector", "may report unreliable magnification."),
        ("Calibration",            "the μm-per-pixel value might be off."),
        ("PanNuke (CellViT)",      "cell area might be off."),
        #("Cell features (Cellpose)", "size related metrics might be off."),
    ]

    def __init__(self, is_ok: bool, pct: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Display scaling")
        self.setModal(True)
        self.setFixedWidth(440)
        self._is_ok = is_ok
        self._pct = pct

        # Force plain light look so the app's dark stylesheet doesn't leak in
        self.setStyleSheet(
            "QDialog { background: #ffffff; }"
            "QLabel { color: #1f2937; background: transparent; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 20)
        layout.setSpacing(14)

        # Title
        title = QLabel(f"Display scaling is {pct}%")
        tf = QFont()
        tf.setPointSize(13)
        tf.setWeight(QFont.Weight.DemiBold)
        title.setFont(tf)
        layout.addWidget(title)

        if is_ok:
            body = QLabel(
                "This is the recommended setting. "
                "Measurements in OnSight will be accurate."
            )
            body.setWordWrap(True)
            layout.addWidget(body)
        else:
            intro = QLabel(
                "OnSight recommends 100%. At other levels, the following "
                "tools may give wrong numbers:"
            )
            intro.setWordWrap(True)
            layout.addWidget(intro)

            # Plain list, no card decorations
            list_box = QFrame()
            list_box.setStyleSheet(
                "QFrame { background: #f7f7f7; border: 1px solid #e3e3e3;"
                "border-radius: 4px; }"
            )
            lv = QVBoxLayout(list_box)
            lv.setContentsMargins(14, 10, 14, 10)
            lv.setSpacing(6)
            for name, desc in self.AFFECTED:
                row = QLabel(f"<b>{name}</b> — {desc}")
                row.setTextFormat(Qt.TextFormat.RichText)
                row.setWordWrap(True)
                row.setStyleSheet("color: #1f2937; background: transparent;")
                lv.addWidget(row)
            layout.addWidget(list_box)

            steps = QLabel(
                "<p style='margin:0; line-height:1.45;'>"
                "<b>To change it:</b><br>"
                "Open Windows Settings → System → Display → Scale, "
                "choose <b>100%</b>, then restart OnSight."
                "</p>"
            )
            steps.setTextFormat(Qt.TextFormat.RichText)
            steps.setWordWrap(True)
            layout.addWidget(steps)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        if not is_ok:
            btn_open = QPushButton("Open Display Settings")
            btn_open.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_open.setStyleSheet(
                "QPushButton { background:#2e7d4f; color:white; padding:7px 16px;"
                "border:none; border-radius:4px; }"
                "QPushButton:hover { background:#36925b; }"
            )
            btn_open.clicked.connect(self._open_settings_then_prompt_restart)
            btn_row.addWidget(btn_open)

        btn_close = QPushButton("Close")
        btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_close.setStyleSheet(
            "QPushButton { background:#f1f1f1; color:#1f2937; padding:7px 18px;"
            "border:1px solid #cfcfcf; border-radius:4px; }"
            "QPushButton:hover { background:#e6e6e6; }"
        )
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

    # -----------------------------------------------------------------------
    def _open_settings_then_prompt_restart(self):
        try:
            os.startfile("ms-settings:display")
        except Exception:
            return
        # Give Windows a beat to open Settings before our prompt steals focus
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(900, self._prompt_restart)

    def _prompt_restart(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("Restart required")
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setText(
            "After you change the scaling to 100% in Windows Settings, "
            "OnSight needs to restart for the change to take effect."
        )
        btn_restart = msg.addButton("Restart now", QMessageBox.ButtonRole.AcceptRole)
        msg.addButton("Later", QMessageBox.ButtonRole.RejectRole)
        msg.exec()
        if msg.clickedButton() is btn_restart:
            self._restart_app()

    def _restart_app(self):
        import subprocess
        try:
            subprocess.Popen([sys.executable] + sys.argv)
            self.accept()
            QApplication.quit()
        except Exception:
            QMessageBox.warning(
                self, "Restart failed",
                "Could not restart OnSight automatically. "
                "Please close and reopen the app manually."
            )