from PyQt6.QtWidgets import QComboBox
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtCore import Qt

# >>> [新增代码 1] 自定义带勾选框的下拉菜单 >>>
class CheckableComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._changed = False
        self.view().pressed.connect(self.handleItemPressed)
        self.setModel(QStandardItemModel(self))

    def handleItemPressed(self, index):
        item = self.model().itemFromIndex(index)
        if item.checkState() == Qt.CheckState.Checked:
            item.setCheckState(Qt.CheckState.Unchecked)
        else:
            item.setCheckState(Qt.CheckState.Checked)
        self._changed = True

    def hidePopup(self):
        # 阻止点击项目时下拉菜单自动收起，方便用户连续打勾
        if not self._changed:
            super().hidePopup()
        self._changed = False

    def get_checked_items(self):
        # 遍历获取所有被打勾的项
        checkedItems = []
        for i in range(self.count()):
            item = self.model().item(i, 0)
            if item.checkState() == Qt.CheckState.Checked:
                checkedItems.append(item.text())
        return checkedItems

    def addItems(self, items, checked_by_default=True):
        for text in items:
            item = QStandardItem(text)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            # 默认全部打勾
            state = Qt.CheckState.Checked if checked_by_default else Qt.CheckState.Unchecked
            item.setData(state, Qt.ItemDataRole.CheckStateRole)
            self.model().appendRow(item)
# <<< [新增结束] <<<