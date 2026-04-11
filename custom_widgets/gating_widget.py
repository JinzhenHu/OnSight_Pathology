# custom_widgets/gating_widget.py
import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QComboBox, 
                             QPushButton, QLabel, QDialogButtonBox, QMessageBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QInputDialog

class HierarchicalGatingDialog(QDialog):
    def __init__(self, df: pd.DataFrame, features: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Interactive Manual Gating (Flow Cytometry Style)")
        self.resize(900, 700)
        self.setStyleSheet("background-color: #ffffff; color: #2c3e50; font-size: 11pt;")
        
        self.raw_df = df.copy()
        # 初始所有细胞都在 "All Cells" (ID = 0) 这个群落里
        self.raw_df['Cluster'] = 0 
        self.features = features
        
        self.cluster_names = {0: "All Cells"}
        self.next_id = 1
        
        self.init_ui()
        self.update_plot()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        
        # --- 左侧控制面板 ---
        side_panel = QVBoxLayout()
        
        side_panel.addWidget(QLabel("<b>1. Gate from (Parent):</b>"))
        self.cb_parent = QComboBox()
        self.refresh_parent_list()
        self.cb_parent.currentIndexChanged.connect(self.update_plot)
        side_panel.addWidget(self.cb_parent)
        
        side_panel.addSpacing(15)
        side_panel.addWidget(QLabel("<b>2. X Axis Feature:</b>"))
        self.cb_x = QComboBox()
        self.cb_x.addItems(self.features)
        self.cb_x.currentTextChanged.connect(self.update_plot)
        side_panel.addWidget(self.cb_x)
        
        side_panel.addWidget(QLabel("<b>3. Y Axis Feature:</b>"))
        self.cb_y = QComboBox()
        self.cb_y.addItems(self.features)
        if len(self.features) > 1: self.cb_y.setCurrentIndex(1)
        self.cb_y.currentTextChanged.connect(self.update_plot)
        side_panel.addWidget(self.cb_y)
        
        side_panel.addStretch()
        
        self.btn_reset = QPushButton("Reset All Gates")
        self.btn_reset.setStyleSheet("background-color: #e74c3c; color: white; padding: 8px; border-radius: 4px;")
        self.btn_reset.clicked.connect(self.reset_all)
        side_panel.addWidget(self.btn_reset)
        
        main_layout.addLayout(side_panel, 1)
        
        # --- 右侧画图区域 ---
        plot_layout = QVBoxLayout()
        self.fig, self.ax = plt.subplots()
        self.fig.patch.set_facecolor('#ffffff')
        self.canvas = FigureCanvas(self.fig)
        plot_layout.addWidget(self.canvas)
        
        # 激活鼠标套索工具
        self.lasso = LassoSelector(self.ax, self.on_select, props={'color': '#e74c3c', 'linewidth': 2})
        
        # 底部确认按钮
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        plot_layout.addWidget(btn_box)
        
        main_layout.addLayout(plot_layout, 3)

    def refresh_parent_list(self):
        self.cb_parent.blockSignals(True)
        self.cb_parent.clear()
        for cid, name in self.cluster_names.items():
            count = (self.raw_df['Cluster'] == cid).sum()
            if count > 0:
                self.cb_parent.addItem(f"{name} (n={count})", cid)
        self.cb_parent.blockSignals(False)

    def update_plot(self):
        self.ax.clear()
        parent_id = self.cb_parent.currentData()
        if parent_id is None: parent_id = 0
        
        x_feat = self.cb_x.currentText()
        y_feat = self.cb_y.currentText()
        
        mask = self.raw_df['Cluster'] == parent_id
        self.current_subset_indices = self.raw_df[mask].index
        
        x_data = self.raw_df.loc[mask, x_feat]
        y_data = self.raw_df.loc[mask, y_feat]
        
        # 绘制当前选中的父群落
        self.scatter = self.ax.scatter(x_data, y_data, s=25, alpha=0.6, c='#3498db', edgecolors='none')
        self.ax.set_xlabel(x_feat)
        self.ax.set_ylabel(y_feat)
        self.ax.set_title(f"Draw a loop to select a sub-population from '{self.cluster_names[parent_id]}'")
        self.ax.grid(True, linestyle='--', alpha=0.5)
        
        self.fig.tight_layout()
        self.canvas.draw()

    def on_select(self, verts):
        path = Path(verts)
        if not hasattr(self, 'scatter'): return
        
        offsets = self.scatter.get_offsets()
        if len(offsets) == 0: return
        
        # 找出落在圈内的点的局部索引
        ind = np.nonzero(path.contains_points(offsets))[0]
        
        if len(ind) > 0:
            name, ok = QInputDialog.getText(self, "New Sub-population", f"Selected {len(ind)} cells. Name this cluster (e.g. Neurons):")
            if ok and name.strip():
                actual_df_indices = self.current_subset_indices[ind]
                self.raw_df.loc[actual_df_indices, 'Cluster'] = self.next_id
                self.cluster_names[self.next_id] = name.strip()
                self.next_id += 1
                
                self.refresh_parent_list()
                self.update_plot()

    def reset_all(self):
        self.raw_df['Cluster'] = 0
        self.cluster_names = {0: "All Cells"}
        self.next_id = 1
        self.refresh_parent_list()
        self.update_plot()