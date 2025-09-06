from PyQt6.QtWidgets import (
    QGroupBox, QVBoxLayout, QPushButton, QWidget, QStyleOptionGroupBox, QStyle
)
from PyQt6.QtCore import Qt


class CollapsibleGroupBox(QGroupBox):
    def __init__(self, title="", parent=None):
        super().__init__(title, parent)

        # --- Toggle button ---
        self._toggle_btn = QPushButton("Hide", self)
        self._toggle_btn.setFixedWidth(60)
        self._toggle_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # prevent stealing focus
        self._toggle_btn.clicked.connect(self._on_toggle_clicked)

        self._toggle_btn.setFlat(True)  # removes 3D frame
        self._toggle_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._toggle_btn.setStyleSheet("""
            QPushButton {
                padding: 0px;
                background: transparent;
                color: gray;
            }
            QPushButton:hover {
                color: black;
                text-decoration: underline;
            }
        """)

        # --- Collapsible content ---
        self._content = QWidget()
        self._layout = QVBoxLayout(self._content)
        self._layout.setContentsMargins(0, 0, 0, 0)

        outer_layout = QVBoxLayout(self)
        outer_layout.addWidget(self._content)

        self._expanded = True

    def content_layout(self):
        return self._layout

    def resizeEvent(self, event):
        """Reposition the toggle button in the title bar area (aligned right)."""
        super().resizeEvent(event)

        opt = QStyleOptionGroupBox()
        self.initStyleOption(opt)

        # Get the rectangle of the title text
        label_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_GroupBox,
            opt,
            QStyle.SubControl.SC_GroupBoxLabel,
            self,
        )

        # Align button with label's vertical center, but flush to the right edge
        x = self.rect().right() - self._toggle_btn.width() + 7
        y = label_rect.center().y() - self._toggle_btn.height() // 2
        self._toggle_btn.move(x, y)

    def _on_toggle_clicked(self):
        self._expanded = not self._expanded
        self._content.setVisible(self._expanded)
        self._toggle_btn.setText("Hide" if self._expanded else "Show")
        self.updateGeometry()
        self.window().adjustSize()
