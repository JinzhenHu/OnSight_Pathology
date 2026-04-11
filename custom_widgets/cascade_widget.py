# custom_widgets/cascade_widget.py
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QComboBox, 
                             QPushButton, QLabel, QSpinBox, QFrame, QCheckBox)
from PyQt6.QtCore import pyqtSignal, Qt
from custom_widgets.CheckableComboBox import CheckableComboBox

class CascadeRuleWidget(QFrame):
    rule_changed = pyqtSignal()
    
    def __init__(self, step_id, available_targets, available_features, parent=None):
        super().__init__(parent)
        self.step_id = step_id
        # --- 白底卡片风格，紧凑边距 ---
        self.setStyleSheet("""
            CascadeRuleWidget {
                background-color: #FFFFFF;
                border: 1px solid #DEE2E6;
                border-radius: 8px;
                margin-bottom: 2px;
            }
            QLabel { color: #2C3E50; font-family: 'Segoe UI'; font-size: 10pt; }
            QComboBox, QSpinBox {
                background-color: #F8F9FA;
                border: 1px solid #CED4DA;
                border-radius: 4px;
                padding: 2px;
                color: #2C3E50;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6) 
        layout.setSpacing(4) 
        
        # 第一行：标题与 Target
        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        title_lbl = QLabel(f"<b>LAYER {step_id}</b>")
        title_lbl.setStyleSheet("color: #339AF0;")
        row1.addWidget(title_lbl)
        
        row1.addStretch()
        row1.addWidget(QLabel("Target:"))
        self.cb_target = QComboBox()
        self.cb_target.addItems(available_targets)
        self.cb_target.currentTextChanged.connect(self.rule_changed.emit)
        row1.addWidget(self.cb_target, 2)
        layout.addLayout(row1)
        
        # 第二行：K 值与特征全选控制
        row2 = QHBoxLayout()
        row2.setContentsMargins(0, 0, 0, 0)
        row2.addWidget(QLabel("Clusters (K):"))
        self.spin_k = QSpinBox()
        self.spin_k.setRange(2, 10)
        self.spin_k.setValue(2)
        self.spin_k.valueChanged.connect(self.rule_changed.emit)
        row2.addWidget(self.spin_k)
        
        row2.addStretch()
        # 全选/全不选 魔法开关
        self.btn_toggle_all = QPushButton("Select All/None Features")
        self.btn_toggle_all.setCheckable(True)
        self.btn_toggle_all.setChecked(True)
        self.btn_toggle_all.setStyleSheet("""
            QPushButton { 
                font-size: 8pt; background-color: #E9ECEF; border-radius: 4px; padding: 2px 5px; 
            }
            QPushButton:checked { background-color: #D0EBFF; color: #1971C2; }
        """)
        self.btn_toggle_all.clicked.connect(self.toggle_features)
        row2.addWidget(self.btn_toggle_all)
        layout.addLayout(row2)
        
        # 第三行：特征选择
        self.cb_features = CheckableComboBox()
        self.cb_features.addItems([str(f) for f in available_features], checked_by_default=True)
        self.cb_features.model().dataChanged.connect(self.rule_changed.emit)
        layout.addWidget(self.cb_features)

    def toggle_features(self, checked):
        """🚀 完美修复：既能更新文字，又不会卡死电脑"""
        model = self.cb_features.model()
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        
        # 1. 临时断开连向后台的 rule_changed 信号，防止被连续触发 14 次
        try:
            model.dataChanged.disconnect(self.rule_changed.emit)
        except Exception:
            pass

        # 2. 正常修改所有打勾状态（此时 CheckableComboBox 内部的文字会自然跟随更新！）
        for i in range(model.rowCount()):
            item = model.item(i)
            if item.isCheckable():
                item.setCheckState(state)

        # 3. 把线接回来
        model.dataChanged.connect(self.rule_changed.emit)
        
        # 4. 手动发送一次请求给后台，让它更新图像
        self.rule_changed.emit()

    def get_config(self):
        return {
            "step_id": self.step_id,
            "target": self.cb_target.currentText(),
            "features": self.cb_features.get_checked_items(),
            "k": self.spin_k.value()
        }

class CascadeClusteringWidget(QWidget):
    pipeline_updated = pyqtSignal(list)
    
    def __init__(self, available_features, parent=None):
        super().__init__(parent)
        self.available_features = available_features
        self.rules = []
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(4)
        
        self.rules_container = QVBoxLayout()
        self.rules_container.setSpacing(4)
        self.layout.addLayout(self.rules_container)
        
        # 底部操作按钮
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 2, 0, 0)
        
        self.btn_add = QPushButton("+ Add Layer")
        self.btn_add.setStyleSheet("""
            QPushButton {
                background-color: #20C997; color: white; border-radius: 6px; 
                padding: 6px; font-weight: bold;
            }
            QPushButton:hover { background-color: #12B886; }
        """)
        self.btn_add.clicked.connect(self.add_rule)
        
        self.btn_remove = QPushButton("✖ Remove Last")
        self.btn_remove.setStyleSheet("""
            QPushButton {
                background-color: #FF8787; color: white; border-radius: 6px; 
                padding: 6px; font-weight: bold;
            }
            QPushButton:hover { background-color: #FA5252; }
        """)
        self.btn_remove.clicked.connect(self.remove_last_rule)
        
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_remove)
        self.layout.addLayout(btn_layout)
        
        self.add_rule()

    def get_available_targets(self):
        leaves = ["All Cells"]
        for rule in self.rules:
            cfg = rule.get_config()
            if cfg["target"] in leaves:
                leaves.remove(cfg["target"])
                for i in range(cfg["k"]):
                    leaves.append(f"S{cfg['step_id']}-C{i}")
        return leaves

    def add_rule(self):
        targets = self.get_available_targets()
        if not targets: return
        
        step_id = len(self.rules) + 1
        rule_w = CascadeRuleWidget(step_id, targets, self.available_features)
        rule_w.rule_changed.connect(self.emit_pipeline)
        
        self.rules.append(rule_w)
        self.rules_container.addWidget(rule_w)
        self.emit_pipeline()

    def remove_last_rule(self):
        if len(self.rules) > 1:
            w = self.rules.pop()
            w.setParent(None)
            w.deleteLater()
            self.emit_pipeline()

    def emit_pipeline(self):
        self.pipeline_updated.emit([r.get_config() for r in self.rules])