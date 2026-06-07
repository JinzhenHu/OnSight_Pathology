# custom_widgets/WelcomeDialog.py
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QFrame, QDialogButtonBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QIcon, QDesktopServices
from PyQt6.QtCore import QUrl


class WelcomeDialog(QDialog):
    """
    First-launch onboarding dialog.
    Shown only on the first launch (or until user has run inference once).
    Returns Accepted regardless of how it's closed - it's informational only.
    """

    DEMO_URL = "https://www.youtube.com/playlist?list=PLm4Z2fSXqbcZzA3XylHtF_oQGSQD-0Jrz"  

    def __init__(self, parent=None, icon_path=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to OnSight Pathology")
        self.setModal(True)
        self.setMinimumWidth(520)
        self.setStyleSheet("background-color: #f7f9fb;")

        main = QVBoxLayout(self)
        main.setContentsMargins(28, 24, 28, 22)
        main.setSpacing(14)

        # --- Header: icon + title ---
        header = QHBoxLayout()
        header.setSpacing(14)

        if icon_path:
            icon = QIcon(icon_path)
            pix = icon.pixmap(56, 56)
            ic_lbl = QLabel()
            ic_lbl.setPixmap(pix)
            header.addWidget(ic_lbl)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        t1 = QLabel("<h2 style='margin:0;'>👋 Welcome to OnSight Pathology</h2>")
        t2 = QLabel(
            "<span style='color:#5a6772; font-size:10pt;'>"
            "Real-time AI analysis for digital pathology"
            "</span>"
        )
        title_box.addWidget(t1)
        title_box.addWidget(t2)
        header.addLayout(title_box)
        header.addStretch()
        main.addLayout(header)

        # --- Separator ---
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("background-color: #dde3ea; max-height: 1px;")
        main.addWidget(line)

        # --- Intro ---
        intro = QLabel("Get started in three quick steps:")
        intro.setStyleSheet("font-size: 11pt; color: #2c3e50; margin-top: 4px;")
        main.addWidget(intro)

        # --- 3 steps ---
        main.addLayout(self._step(
            "1",
            "Open your slide viewer",
            "Launch QuPath, your scanner's software, or any image viewer "
            "showing the slide you want to analyze."
        ))
        main.addLayout(self._step(
            "2",
            "Select a screen region",
            'Click <b>"Select Screen Region"</b> in OnSight and choose a '
            "box around the area of the slide you want to analyze."
        ))
        main.addLayout(self._step(
            "3",
            "Choose a model and Start",
            "Pick a model from the <b>\"Model Selection\"</b> dropdown "
            'and click <b>Start</b>. Analysis runs in real time.'
        ))

        # --- Tip ---
        tip = QLabel(
            "<div style='background-color:#eef5ff; border-left:3px solid #339af0; "
            "padding:10px 12px; border-radius:4px; color:#34495e; font-size:10pt;'>"
            "💡 <b>Tip:</b> The first time you use a model, it will download "
            "automatically (a few hundred MB)."
            "</div>"
        )
        tip.setTextFormat(Qt.TextFormat.RichText)
        tip.setWordWrap(True)
        main.addWidget(tip)

        # --- Don't show again ---
        self.chk_dont_show = QCheckBox("Don't show this again")
        self.chk_dont_show.setStyleSheet("color: #5a6772;")
        main.addWidget(self.chk_dont_show)

        # --- Buttons ---
        btn_row = QHBoxLayout()
        btn_demo = QPushButton("🎬  Watch demo")
        btn_demo.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_demo.setStyleSheet(
            "QPushButton { background:#ffffff; color:#339af0; "
            "border:1px solid #339af0; padding:8px 16px; border-radius:6px; }"
            "QPushButton:hover { background:#eef5ff; }"
        )
        btn_demo.clicked.connect(self._open_demo)

        btn_ok = QPushButton("Got it, let's start")
        btn_ok.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_ok.setDefault(True)
        btn_ok.setStyleSheet(
            "QPushButton { background:#339af0; color:white; "
            "border:none; padding:8px 18px; border-radius:6px; font-weight:bold; }"
            "QPushButton:hover { background:#228be6; }"
        )
        btn_ok.clicked.connect(self.accept)

        btn_row.addWidget(btn_demo)
        btn_row.addStretch()
        btn_row.addWidget(btn_ok)
        main.addLayout(btn_row)

    def _step(self, num: str, title: str, body: str) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(12)

        # Numbered circle
        circle = QLabel(num)
        circle.setFixedSize(28, 28)
        circle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        circle.setStyleSheet(
            "background-color:#339af0; color:white; "
            "border-radius:14px; font-weight:bold; font-size:11pt;"
        )

        # Text
        text_box = QVBoxLayout()
        text_box.setSpacing(1)
        t = QLabel(f"<b style='font-size:10.5pt; color:#2c3e50;'>{title}</b>")
        b = QLabel(body)
        b.setStyleSheet("color:#5a6772; font-size:10pt;")
        b.setWordWrap(True)
        b.setTextFormat(Qt.TextFormat.RichText)
        text_box.addWidget(t)
        text_box.addWidget(b)

        row.addWidget(circle, alignment=Qt.AlignmentFlag.AlignTop)
        row.addLayout(text_box, stretch=1)
        return row

    def _open_demo(self):
        QDesktopServices.openUrl(QUrl(self.DEMO_URL))

    def is_dont_show_checked(self) -> bool:
        return self.chk_dont_show.isChecked()