# custom_widgets/cascade_widget.py
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QComboBox, 
                             QPushButton, QLabel, QSpinBox, QFrame, 
                             QStackedWidget, QDoubleSpinBox)
from PyQt6.QtCore import pyqtSignal, Qt
from custom_widgets.CheckableComboBox import CheckableComboBox

class CascadeRuleWidget(QFrame):
    rule_changed = pyqtSignal()
    
    def __init__(self, step_id, available_targets, available_features, parent=None):
        super().__init__(parent)
        self.step_id = step_id
        self.available_features = available_features # 保存一份留给全选/全不选重置用
        
        self.setStyleSheet("""
            CascadeRuleWidget {
                background-color: #FFFFFF;
                border: 1px solid #DEE2E6;
                border-radius: 8px;
                margin-bottom: 2px;
            }
            QLabel { color: #2C3E50; font-family: 'Segoe UI'; font-size: 10pt; }
            QComboBox, QSpinBox, QDoubleSpinBox {
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
        
        # --- 第一行：层级、目标与方法切换 ---
        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        
        title_lbl = QLabel(f"<b>L{step_id}</b>")
        title_lbl.setStyleSheet("color: #339AF0; min-width: 25px;")
        row1.addWidget(title_lbl)
        
        row1.addWidget(QLabel("Tar:"))
        self.cb_target = QComboBox()
        self.cb_target.addItems(available_targets)
        self.cb_target.currentTextChanged.connect(self.rule_changed.emit)
        row1.addWidget(self.cb_target, 2)
        
        row1.addWidget(QLabel("Method:"))
        self.cb_method = QComboBox()
        self.cb_method.addItems(["K-Means", "Threshold"])
        self.cb_method.currentTextChanged.connect(self._on_method_changed)
        row1.addWidget(self.cb_method, 2)
        
        layout.addLayout(row1)
        
        # --- 第二行：动态参数面板 ---
        self.stacked_params = QStackedWidget()
        
        # [卡片 1：K-Means 参数]
        self.page_kmeans = QWidget()
        layout_km = QHBoxLayout(self.page_kmeans)
        layout_km.setContentsMargins(0, 0, 0, 0)
        
        layout_km.addWidget(QLabel("K:"))
        self.spin_k = QSpinBox()
        self.spin_k.setRange(2, 10)
        self.spin_k.setValue(2)
        self.spin_k.valueChanged.connect(self.rule_changed.emit)
        layout_km.addWidget(self.spin_k)
        
        # 全选/全不选 按钮
        self.btn_toggle_all = QPushButton("All/None")
        self.btn_toggle_all.setCheckable(True)
        self.btn_toggle_all.setChecked(False) # 默认不选中
        self.btn_toggle_all.setStyleSheet("QPushButton { font-size: 8pt; background-color: #E9ECEF; border-radius: 4px; padding: 2px 4px; } QPushButton:checked { background-color: #D0EBFF; color: #1971C2; }")
        self.btn_toggle_all.clicked.connect(self.toggle_features)
        layout_km.addWidget(self.btn_toggle_all)
        
        self.cb_features = CheckableComboBox()
        # 🚀 初始时设置 checked_by_default=False，完美对应 0 features selected
        self.cb_features.addItems([str(f) for f in available_features], checked_by_default=False)
        self.cb_features.model().dataChanged.connect(self.rule_changed.emit)
        layout_km.addWidget(self.cb_features, 3)
        
        # [卡片 2：Threshold 参数]
        self.page_thresh = QWidget()
        layout_th = QHBoxLayout(self.page_thresh)
        layout_th.setContentsMargins(0, 0, 0, 0)
        
        layout_th.addWidget(QLabel("Feat:"))
        self.cb_single_feat = QComboBox()
        self.cb_single_feat.addItems([str(f) for f in available_features])
        self.cb_single_feat.currentTextChanged.connect(self.rule_changed.emit)
        layout_th.addWidget(self.cb_single_feat, 2)
        
        layout_th.addWidget(QLabel(" Cutoff:"))
        self.spin_thresh = QDoubleSpinBox()
        self.spin_thresh.setRange(-9999999.0, 9999999.0) 
        self.spin_thresh.setDecimals(4)
        self.spin_thresh.setValue(0.5)
        self.spin_thresh.valueChanged.connect(self.rule_changed.emit)
        layout_th.addWidget(self.spin_thresh, 2)
        
        self.stacked_params.addWidget(self.page_kmeans)
        self.stacked_params.addWidget(self.page_thresh)
        layout.addWidget(self.stacked_params)

    def _on_method_changed(self, method):
        if method == "K-Means":
            self.stacked_params.setCurrentIndex(0)
        else:
            self.stacked_params.setCurrentIndex(1)
        self.rule_changed.emit()

    def toggle_features(self, checked):
        """🚀 完美修复UI不同步问题：利用重建列表强制刷新文本缓存"""
        try: self.cb_features.model().dataChanged.disconnect(self.rule_changed.emit)
        except Exception: pass
        
        self.cb_features.blockSignals(True)
        self.cb_features.clear() # 清空旧元素，这会强制销毁旧的文字缓存
        # 重新加入元素，checked_by_default 完全由 checked 变量控制！
        self.cb_features.addItems([str(f) for f in self.available_features], checked_by_default=checked)
        self.cb_features.blockSignals(False)
        
        # 重新挂载信号，并单次发送到后台
        self.cb_features.model().dataChanged.connect(self.rule_changed.emit)
        self.rule_changed.emit()

    def get_config(self):
        method = self.cb_method.currentText()
        if method == "K-Means":
            return {
                "step_id": self.step_id,
                "target": self.cb_target.currentText(),
                "method": "K-Means",
                "features": self.cb_features.get_checked_items(),
                "k": self.spin_k.value()
            }
        else:
            return {
                "step_id": self.step_id,
                "target": self.cb_target.currentText(),
                "method": "Threshold",
                "feature": self.cb_single_feat.currentText(),
                "threshold": self.spin_thresh.value(),
                "k": 2 
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
        
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 2, 0, 0)
        
        self.btn_add = QPushButton("+ Add Layer")
        self.btn_add.setStyleSheet("QPushButton { background-color: #20C997; color: white; border-radius: 6px; padding: 6px; font-weight: bold; } QPushButton:hover { background-color: #12B886; }")
        self.btn_add.clicked.connect(self.add_rule)
        
        self.btn_remove = QPushButton("✖ Remove Last")
        self.btn_remove.setStyleSheet("QPushButton { background-color: #FF8787; color: white; border-radius: 6px; padding: 6px; font-weight: bold; } QPushButton:hover { background-color: #FA5252; }")
        self.btn_remove.clicked.connect(self.remove_last_rule)
        
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_remove)
        self.layout.addLayout(btn_layout)
        
        self.add_rule()

    # 🚀 族谱树命名生成器 (C1, C2 -> C1.1, C1.2)
    def get_available_targets(self):
        leaves = ["All Cells"]
        for rule in self.rules:
            cfg = rule.get_config()
            t = cfg["target"]
            if t in leaves:
                leaves.remove(t)
                prefix = "C" if t == "All Cells" else f"{t}."
                for i in range(cfg["k"]):
                    leaves.append(f"{prefix}{i+1}")
        return leaves

    def add_rule(self):
        targets = self.get_available_targets()
        if not targets: return
        rule_w = CascadeRuleWidget(len(self.rules) + 1, targets, self.available_features)
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