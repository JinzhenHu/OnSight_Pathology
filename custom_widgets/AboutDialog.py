from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QSpacerItem, QSizePolicy
from PyQt6.QtGui import QPixmap, QIcon
from PyQt6.QtCore import Qt


class AboutDialog(QDialog):
    def __init__(self, parent=None, icon_path=None):
        super().__init__(parent)
        self.setWindowTitle("About")
        self.setModal(True)
        self.setFixedSize(400, 300)

        self.setStyleSheet("background-color: #f0f0f0;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # Icon at the top
        if icon_path:
            icon = QIcon(icon_path)
            pixmap = icon.pixmap(64, 64)  # directly get 64x64, Qt handles scaling cleanly

            # pixmap = QPixmap(icon_path)
            icon_lbl = QLabel()
            # icon_lbl.setPixmap(pixmap.scaled(32, 32, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            icon_lbl.setPixmap(pixmap)
            icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(icon_lbl)

        title_lbl = QLabel("<h2>OnSight</h2>")
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_lbl)

        # Description text with clickable link
        desc_lbl = QLabel()
        desc_lbl.setTextFormat(Qt.TextFormat.RichText)
        desc_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        desc_lbl.setOpenExternalLinks(True)
        desc_lbl.setText(
            "OnSight is a standalone, vendor-agnostic software that bridges the gap "
            "between powerful AI models and the daily workflow of pathologists, offering "
            "real-time analysis without compromising data privacy.<br><br>"
            "Visit: <a href='https://onsightpathology.github.io'>Project Website</a>"
        )
        desc_lbl.setWordWrap(True)
        layout.addWidget(desc_lbl)

        # Spacer to push button down
        layout.addItem(QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        layout.addWidget(ok_btn, alignment=Qt.AlignmentFlag.AlignCenter)
