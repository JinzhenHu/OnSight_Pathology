from PyQt6.QtWidgets import QComboBox, QListView
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QFontMetrics
from PyQt6.QtCore import Qt, QEvent

class CheckableComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setView(QListView(self))
        self.view().pressed.connect(self.handle_item_pressed)
        self.setModel(QStandardItemModel(self))
        
        self._changed = False

    def handle_item_pressed(self, index):
        item = self.model().itemFromIndex(index)
        
        if item.checkState() == Qt.CheckState.Checked:
            item.setCheckState(Qt.CheckState.Unchecked)
        else:
            item.setCheckState(Qt.CheckState.Checked)
            
        self._changed = True
        self._update_display_text()

    def hidePopup(self):
        if not self._changed:
            super().hidePopup()
        self._changed = False

    def addItems(self, items, checked_by_default=False):
        for text in items:
            item = QStandardItem(text)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            
            check_state = Qt.CheckState.Checked if checked_by_default else Qt.CheckState.Unchecked
            item.setData(check_state, Qt.ItemDataRole.CheckStateRole)
            
            self.model().appendRow(item)
            
        self._update_display_text()

    def get_checked_items(self):
        checked = []
        for i in range(self.model().rowCount()):
            item = self.model().item(i)
            if item.checkState() == Qt.CheckState.Checked:
                checked.append(item.text())
        return checked

    def _update_display_text(self):
        checked_items = self.get_checked_items()
        
        if not checked_items:
            self.setPlaceholderText("Select...")
            self.setCurrentIndex(-1)
        else:
            if len(checked_items) == 1:
                display_text = checked_items[0]
            else:
                display_text = f"{len(checked_items)} features selected"
                

            self.setPlaceholderText(display_text)
            self.setCurrentIndex(-1) 

    def paintEvent(self, event):
        super().paintEvent(event)