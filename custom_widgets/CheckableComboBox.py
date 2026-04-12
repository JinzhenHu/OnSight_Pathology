# from PyQt6.QtWidgets import QComboBox, QListView
# from PyQt6.QtGui import QStandardItemModel, QStandardItem, QFontMetrics
# from PyQt6.QtCore import Qt, QEvent

# class CheckableComboBox(QComboBox):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         # 使用 QListView 以支持更好的点击交互
#         self.setView(QListView(self))
#         self.view().pressed.connect(self.handle_item_pressed)
#         self.setModel(QStandardItemModel(self))
        
#         # 记录是否刚刚发生过点击，用来阻止菜单秒关
#         self._changed = False

#     def handle_item_pressed(self, index):
#         """点击整行任意位置即可触发打勾/取消打勾"""
#         item = self.model().itemFromIndex(index)
        
#         # 反转当前的勾选状态
#         if item.checkState() == Qt.CheckState.Checked:
#             item.setCheckState(Qt.CheckState.Unchecked)
#         else:
#             item.setCheckState(Qt.CheckState.Checked)
            
#         self._changed = True
#         self._update_display_text()

#     def hidePopup(self):
#         """拦截默认的关闭行为：如果是点击了选项，则保持展开；如果是点击了外部，才关闭"""
#         if not self._changed:
#             super().hidePopup()
#         self._changed = False

#     def addItems(self, items, checked_by_default=False):
#         """兼容 app.py 的数据加载接口"""
#         for text in items:
#             item = QStandardItem(text)
#             # 开启可用性和复选框属性
#             item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            
#             # 设置初始状态
#             check_state = Qt.CheckState.Checked if checked_by_default else Qt.CheckState.Unchecked
#             item.setData(check_state, Qt.ItemDataRole.CheckStateRole)
            
#             self.model().appendRow(item)
            
#         self._update_display_text()

#     def get_checked_items(self):
#         """返回所有打了勾的文本列表"""
#         checked = []
#         for i in range(self.model().rowCount()):
#             item = self.model().item(i)
#             if item.checkState() == Qt.CheckState.Checked:
#                 checked.append(item.text())
#         return checked

#     def _update_display_text(self):
#         """让下拉菜单折叠时，显示 '已选择 X 项' 或选中的具体名字"""
#         checked_items = self.get_checked_items()
        
#         if not checked_items:
#             self.setPlaceholderText("Select features...")
#             self.setCurrentIndex(-1)
#         else:
#             # 如果只选了 1 个，直接显示名字；选了多个，显示数量
#             if len(checked_items) == 1:
#                 display_text = checked_items[0]
#             else:
#                 display_text = f"{len(checked_items)} features selected"
                
#             # 设置提示词
#             self.setPlaceholderText(display_text)
#             self.setCurrentIndex(-1) 

#     # 绘制事件拦截（让 PlaceholderText 能正常显示在 QComboBox 上）
#     def paintEvent(self, event):
#         super().paintEvent(event)

# custom_widgets/CheckableComboBox.py
# custom_widgets/CheckableComboBox.py
# custom_widgets/CheckableComboBox.py
from PyQt6.QtWidgets import QComboBox, QListView
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtCore import Qt

class CheckableComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setView(QListView(self))
        self.view().pressed.connect(self.handle_item_pressed)
        self.setModel(QStandardItemModel(self))
        self._changed = False
        
        # 默认基准词
        self._placeholder_base = "features"

    @property
    def placeholder_base(self):
        return self._placeholder_base

    @placeholder_base.setter
    def placeholder_base(self, value):
        self._placeholder_base = value
        self._update_display_text()

    def handle_item_pressed(self, index):
        item = self.model().itemFromIndex(index)
        if item:
            new_state = Qt.CheckState.Unchecked if item.checkState() == Qt.CheckState.Checked else Qt.CheckState.Checked
            item.setCheckState(new_state)
            self._changed = True
            self._update_display_text()

    def hidePopup(self):
        if not self._changed:
            super().hidePopup()
        self._changed = False

    def clear_items(self):
        """🚀 绝对安全的清空方法：只清空数据，不摧毁模型结构！"""
        self.model().clear()
        self._update_display_text()

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
            if item and item.checkState() == Qt.CheckState.Checked:
                checked.append(item.text())
        return checked

    def _update_display_text(self):
        """动态显示 'Select groups...' 等文字"""
        checked_items = self.get_checked_items()
        if not checked_items:
            self.setPlaceholderText(f"Select {self._placeholder_base}...")
        else:
            if len(checked_items) == 1:
                display_text = checked_items[0]
            else:
                display_text = f"{len(checked_items)} {self._placeholder_base} selected"
            self.setPlaceholderText(display_text)
        self.setCurrentIndex(-1) 

    def paintEvent(self, event):
        super().paintEvent(event)