from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton,
    QHeaderView, QSizePolicy
)
from PyQt6.QtCore import Qt

class TablePopup(QDialog):
    def __init__(self, parent=None, items=None, title="Table Popup"):
        super().__init__(parent)
        self.setWindowTitle(title)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        table = QTableWidget(self)

        if items is None:
            items = []

        if len(items) % 2 != 0:
            items.append("")

        rows = len(items) // 2
        cols = 2

        table.setRowCount(rows)
        table.setColumnCount(cols)

        table.horizontalHeader().setVisible(False)
        table.verticalHeader().setVisible(False)

        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

        for i in range(rows):
            for j in range(cols):
                idx = i * 2 + j
                item = QTableWidgetItem(str(items[idx]))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # Make cell read-only
                table.setItem(i, j, item)

        layout.addWidget(table)

        ok_button = QPushButton("OK", self)
        ok_button.clicked.connect(self.accept)
        layout.addWidget(ok_button)

        self.setLayout(layout)

        # === Adjust height to fit content ===
        table_height = table.horizontalHeader().height()
        for i in range(table.rowCount()):
            table_height += table.rowHeight(i)

        button_height = ok_button.sizeHint().height()
        total_height = table_height + button_height + 20 + 20
        self.setFixedHeight(total_height)

        # === Adjust width to fit columns ===
        table.resizeColumnsToContents()
        total_width = table.verticalHeader().width()
        for i in range(table.columnCount()):
            total_width += table.columnWidth(i)
        total_width += 4 + 20  # account for border/padding
        self.setFixedWidth(total_width)
