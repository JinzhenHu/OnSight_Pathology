from PyQt6.QtWidgets import (
    QVBoxLayout, QLabel, QHBoxLayout, QFrame, QDialog,
    QCheckBox, QStyle, QDialogButtonBox
)
from PyQt6.QtCore import Qt


class DisclaimerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Disclaimer")
        self.setModal(True)
        self.setMinimumWidth(450)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # --- Top section with icon and text ---
        top_layout = QHBoxLayout()
        top_layout.setSpacing(15)

        # Icon in its own vbox to center it vertically
        icon_layout = QVBoxLayout()
        icon_layout.addStretch()

        icon_label = QLabel()
        # Use a standard Qt icon for a professional look
        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning)
        icon_label.setPixmap(icon.pixmap(48, 48))
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # horizontal center
        icon_layout.addWidget(icon_label)

        icon_layout.addStretch()

        top_layout.addLayout(icon_layout)  # add the vbox to the left

        # Text section (Title + Message)
        text_layout = QVBoxLayout()

        title_label = QLabel("<b>Important Notice</b>")
        font = title_label.font()
        font.setPointSize(12)
        title_label.setFont(font)
        text_layout.addWidget(title_label)

        message = QLabel(
            "This tool is intended for research and educational purposes only. "
            "It is not has not yet been validated for clinical diagnostic use."
        )
        message.setWordWrap(True)
        text_layout.addWidget(message)

        top_layout.addLayout(text_layout)
        main_layout.addLayout(top_layout)

        # --- Separator Line ---
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFixedHeight(1)
        line.setStyleSheet("background-color: #4a627a;")  # A color from the theme
        main_layout.addWidget(line)

        # --- Checkbox ---
        self.checkbox = QCheckBox("Don't show this message again")
        main_layout.addWidget(self.checkbox, alignment=Qt.AlignmentFlag.AlignCenter)

        # --- OK Button ---
        # ok_button = QPushButton("OK")
        # ok_button.clicked.connect(self.accept)
        # ok_button.setDefault(True) # Allows pressing Enter to accept
        # ok_button.setMinimumWidth(120)
        #
        # button_layout = QHBoxLayout()
        # button_layout.addStretch()
        # button_layout.addWidget(ok_button)
        # button_layout.addStretch()

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)
