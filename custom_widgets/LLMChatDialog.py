import sys
import time
import json
import numpy as np
import gc
import os

import torch
from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QDialog, QTextEdit, QWidget, QFrame, QGraphicsDropShadowEffect
)
from PyQt6.QtGui import QPixmap, QImage, QTextCursor, QColor, QFont
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QProcess
from PIL import Image
import tempfile

from utils import resource_path

class LLMChatDialog(QDialog):
    """Modern, sleek, white-themed chat interface for the LLM."""

    def __init__(self, frame_rgb: np.ndarray, llm_metadata_path, llm_precision, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI Copilot")
        self.resize(500, 750)
        
        # ==========================================
        # 1. 酷炫的全局亮色主题 (Apple / ChatGPT 风格)
        # ==========================================
        self.setStyleSheet("""
            QDialog {
                background-color: #FFFFFF; /* 纯白底色 */
            }
            QTextEdit {
                background-color: #FFFFFF;
                border: none;
                font-family: 'Segoe UI', -apple-system, sans-serif;
                font-size: 11pt;
                color: #2C3E50;
            }
            /* 现代化的圆角输入框 */
            QLineEdit {
                background-color: #F4F6F8;
                border: 1px solid #E9ECEF;
                border-radius: 20px;
                padding: 10px 18px;
                font-size: 11pt;
                color: #2C3E50;
            }
            QLineEdit:focus {
                border: 1px solid #339AF0;
                background-color: #FFFFFF;
            }
            /* 圆角发送按钮 */
            QPushButton#SendBtn {
                background-color: #339AF0;
                color: white;
                border-radius: 20px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 11pt;
            }
            QPushButton#SendBtn:hover {
                background-color: #228BE6;
            }
            QPushButton#SendBtn:disabled {
                background-color: #A5D8FF;
            }
            /* 酷炫的 Update 图片按钮 */
            QPushButton#UpdateBtn {
                background-color: rgba(255, 255, 255, 200);
                color: #495057;
                border: 1px solid #DEE2E6;
                border-radius: 15px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton#UpdateBtn:hover {
                background-color: #F8F9FA;
                color: #228BE6;
                border: 1px solid #228BE6;
            }
        """)

        self.model_ready = False
        self.frame_rgb = frame_rgb
        self.parent = parent
        self.msgs = []

        self.llm_metadata_path = llm_metadata_path
        self.llm_precision = llm_precision

        # Save image as temporary PNG
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            self.temp_img_path = f.name
            Image.fromarray(frame_rgb).save(self.temp_img_path)

        # ==========================================
        # 2. GUI 布局重构
        # ==========================================
        main_vbox = QVBoxLayout(self)
        main_vbox.setContentsMargins(20, 20, 20, 20)
        main_vbox.setSpacing(15)

        # --- 顶部：图片预览区 (带悬浮感和按钮) ---
        img_container = QWidget()
        img_layout = QVBoxLayout(img_container)
        img_layout.setContentsMargins(0, 0, 0, 0)
        
        self.lbl_img = QLabel()
        self.lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_img.setStyleSheet("border-radius: 10px; background-color: #F8F9FA;")
        self._set_preview_image(self.frame_rgb)
        
        # Update 按钮悬浮在图片下方
        self.btn_update = QPushButton("📸 Refresh Vision")
        self.btn_update.setObjectName("UpdateBtn")
        self.btn_update.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_update.clicked.connect(self.on_update_image)
        
        # 将图片和按钮居中摆放
        h_img_center = QHBoxLayout()
        h_img_center.addStretch()
        h_img_center.addWidget(self.lbl_img)
        h_img_center.addStretch()
        
        h_btn_center = QHBoxLayout()
        h_btn_center.addStretch()
        h_btn_center.addWidget(self.btn_update)
        h_btn_center.addStretch()

        img_layout.addLayout(h_img_center)
        img_layout.addLayout(h_btn_center)
        main_vbox.addWidget(img_container)

        # --- 中间：无边框优雅聊天流 ---
        self.txt_history = QTextEdit()
        self.txt_history.setReadOnly(True)
        # 隐藏滚动条，让它更像网页
        self.txt_history.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        main_vbox.addWidget(self.txt_history)

        # --- 底部：现代化输入区 ---
        input_layout = QHBoxLayout()
        input_layout.setSpacing(10)
        
        self.inp = QLineEdit()
        self.inp.setPlaceholderText("Message AI Copilot...")
        self.inp.returnPressed.connect(self.on_send) # 按回车也能发
        
        self.btn_send = QPushButton("Send")
        self.btn_send.setObjectName("SendBtn")
        self.btn_send.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_send.clicked.connect(self.on_send)
        
        input_layout.addWidget(self.inp)
        input_layout.addWidget(self.btn_send)
        main_vbox.addLayout(input_layout)

        # 初始欢迎语
        welcome_html = """
        <div style='text-align: center; margin-top: 20px;'>
            <span style='color: #ADB5BD; font-size: 10pt;'>
                🚀 Initializing AI engine... This may take a moment.
            </span>
        </div>
        """
        self.txt_history.setHtml(welcome_html)
        self.inp.setEnabled(False)
        self.btn_send.setEnabled(False)

        # Launch subprocess
        self._start_process()

    def _set_preview_image(self, frame_rgb):
        """Helper function to set and scale the preview image smoothly."""
        h, w, c = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, c * w, QImage.Format.Format_RGB888)
        # 添加一点圆角效果 (由外层 QLabel 的样式控制)
        pixmap = QPixmap.fromImage(qimg).scaled(220, 220, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.lbl_img.setPixmap(pixmap)

    def on_update_image(self):
        if not self.parent.thread or not self.parent.thread.isRunning():
            from PyQt6.QtWidgets import QMessageBox
            # 自定义弹窗样式
            msg = QMessageBox(self)
            msg.setWindowTitle("Tracker Paused")
            msg.setText("Real-time tracking is currently stopped.\nPlease start it in the main window to grab a new frame.")
            msg.setStyleSheet("background-color: white; color: black;")
            msg.exec()
            return
            
        if getattr(self.parent, 'latest_frame', None) is None:
            return
            
        new_frame = self.parent.latest_frame.copy()
        self.frame_rgb = new_frame
        self._set_preview_image(self.frame_rgb)
        
        Image.fromarray(new_frame).save(self.temp_img_path)
        
        if self.process and self.process.state() == QProcess.ProcessState.Running:
            msg = json.dumps({"type": "update_image", "path": self.temp_img_path}) + "\n"
            self.process.write(msg.encode("utf-8"))
            
        # 优雅的灰色居中系统提示
        sys_html = """
        <div style="text-align: center; margin: 15px 0;">
            <span style="background-color: #F8F9FA; color: #868E96; padding: 4px 12px; border-radius: 12px; font-size: 9pt;">
                👁️ Vision synchronized with current microscope view
            </span>
        </div>
        """
        self.txt_history.append(sys_html)


    def _start_process(self):
        self.process = QProcess(self)

        if not self.llm_precision:
            self.llm_precision = "8bit"

        if getattr(sys, 'frozen', False):
            script_path = os.path.join(os.path.dirname(sys.executable), "llm_worker_process.exe")
            self.process.setProgram(script_path)
            self.process.setArguments(["--cfg", self.llm_metadata_path, "--image", self.temp_img_path, "--precision", self.llm_precision])
        else:
            script_path = resource_path("llm_worker_process.py")
            self.process.setProgram(sys.executable)
            self.process.setArguments([script_path, "--cfg", self.llm_metadata_path, "--image", self.temp_img_path, "--precision", self.llm_precision])

        self.process.readyReadStandardOutput.connect(self._handle_stdout)
        self.process.start()

    def _handle_stdout(self):
        output = bytes(self.process.readAllStandardOutput()).decode("utf-8")
        for line in output.strip().splitlines():
            try:
                msg = json.loads(line)
                msg_type = msg.get("type")

                if msg_type == "ready":
                    self.model_ready = True
                    self.txt_history.clear()
                    self.txt_history.append("""
                    <div style='text-align: center; margin-top: 10px; margin-bottom: 20px;'>
                        <span style='color: #20C997; font-weight: bold; font-size: 10pt;'>
                            ✨ Model Ready. How can I help you?
                        </span>
                    </div>
                    """)
                    self.inp.setEnabled(True)
                    self.btn_send.setEnabled(True)
                    self.inp.setFocus()

                elif msg_type == "chunk":
                    # 接收流式文本
                    self.txt_history.moveCursor(QTextCursor.MoveOperation.End)
                    self.txt_history.insertPlainText(msg["text"])
                    # 自动滚到底部
                    self.txt_history.verticalScrollBar().setValue(self.txt_history.verticalScrollBar().maximum())

                elif msg_type == "stream_end":
                    self.inp.setEnabled(True)
                    self.btn_send.setEnabled(True)
                    self.inp.setFocus()
                    self.txt_history.append("<br>") # 留点呼吸空间

                elif msg_type == "error":
                    self.txt_history.append(f"<b style='color:#FA5252;'>Error:</b> {msg['text']}<br>")
                    self.inp.setEnabled(True)
                    self.btn_send.setEnabled(True)

            except Exception as e:
                pass

    def on_send(self):
        text = self.inp.text().strip()
        if not self.model_ready or not text:
            return

        self.inp.clear()

        # 1. 用户的炫酷蓝色气泡 (右对齐)
        user_html = f"""
        <div style="text-align: right; margin-top: 15px; margin-bottom: 5px;">
            <span style="background-color: #339AF0; color: white; padding: 10px 16px; border-radius: 16px; font-size: 11pt; display: inline-block;">
                {text}
            </span>
        </div>
        """
        self.txt_history.append(user_html)

        # 2. AI 的简洁灰色前缀 (左对齐)
        ai_start_html = """
        <div style="margin-top: 10px; margin-bottom: 5px;">
            <b style="color: #20C997; font-size: 11pt;">✦ Assistant</b>
        </div>
        """
        self.txt_history.append(ai_start_html)
        
        self.inp.setEnabled(False)
        self.btn_send.setEnabled(False)

        if self.process:
            msg = json.dumps({"type": "prompt", "text": text}) + "\n"
            self.process.write(msg.encode("utf-8"))

    def closeEvent(self, event):
        if self.process and self.process.state() == QProcess.ProcessState.Running:
            self.process.kill()
            self.process.finished.connect(self._cleanup_temp_file)
            self.process.waitForFinished()  
        else:
            self._cleanup_temp_file()
        super().closeEvent(event)

    def _cleanup_temp_file(self):
        if os.path.exists(self.temp_img_path):
            try:
                os.remove(self.temp_img_path)
            except Exception:
                pass