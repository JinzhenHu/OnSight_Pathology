# Custom LLM Chat Dialog with sleek white theme for OnSight Pathology
# Unified for Windows and macOS — platform differences are handled inline.
import sys
import time
import json
import numpy as np
import gc
import os
import torch
from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QDialog,
    QTextEdit, QWidget, QFrame, QGraphicsDropShadowEffect, QProgressBar
)
from PyQt6.QtGui import QPixmap, QImage, QTextCursor, QColor, QFont
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QProcess
from PIL import Image
import tempfile

from utils import resource_path


_IS_MAC = sys.platform == "darwin"


class LLMChatDialog(QDialog):
    """Modern, sleek, white-themed chat interface for the LLM."""

    def __init__(self, frame_rgb: np.ndarray, llm_metadata_path, llm_precision, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI Copilot")
        self.resize(500, 750)
        self.BASE_FONT_SIZE = 12
        # =================================================================================
        # 1. cool style sheet for the entire dialog
        # ====================================================================
        self.setStyleSheet("""
            QDialog {
                background-color: #FFFFFF; /* pure white background */
            }
            QTextEdit {
                background-color: #FFFFFF;
                border: none;
                font-family: 'Segoe UI', -apple-system, sans-serif;
                font-size: 13pt;
                color: #2C3E50;
            }
            /* Modern rounded input field */
            QLineEdit {
                background-color: #F4F6F8;
                border: 1px solid #E9ECEF;
                border-radius: 20px;
                padding: 10px 18px;
                font-size: 13pt;
                color: #2C3E50;
            }
            QLineEdit:focus {
                border: 1px solid #339AF0;
                background-color: #FFFFFF;
            }
            QPushButton#SendBtn {
                background-color: #339AF0;
                color: white;
                border: none;
                border-radius: 20px;
                padding: 10px 24px;
                font-weight: bold;
                font-size: 13pt;
            }
            QPushButton#SendBtn:hover { background-color: #228BE6; }
            QPushButton#SendBtn:disabled { background-color: #ADB5BD; }
            QPushButton#UpdateBtn {
                background-color: white;
                color: #495057;
                border: 1px solid #E9ECEF;
                border-radius: 16px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 13pt;
            }
            QPushButton#UpdateBtn:hover {
                background-color: #F8F9FA;
                color: #228BE6;
                border: 1px solid #228BE6;
            }
        """)

        self.model_ready = False
        self._error_shown = False
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
        # 2. GUI layout reconstruction
        # ==========================================
        main_vbox = QVBoxLayout(self)
        main_vbox.setContentsMargins(20, 20, 20, 20)
        main_vbox.setSpacing(15)

        # --- Top: Image preview area (with floating effect and button) ---
        img_container = QWidget()
        img_layout = QVBoxLayout(img_container)
        img_layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_img = QLabel()
        self.lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_img.setStyleSheet("border-radius: 10px; background-color: #F8F9FA;")
        self._set_preview_image(self.frame_rgb)

        # Update button floating below the image
        self.btn_update = QPushButton("📸 Refresh Vision")
        self.btn_update.setObjectName("UpdateBtn")
        self.btn_update.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_update.setAutoDefault(False)
        self.btn_update.clicked.connect(self.on_update_image)

        # Center the image and button
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

        # --- Loading progress UI (visible until model is ready) ---
        self.loading_container = QWidget()
        loading_v = QVBoxLayout(self.loading_container)
        loading_v.setContentsMargins(40, 20, 40, 10)
        loading_v.setSpacing(10)

        self.lbl_loading_title = QLabel("🚀 Initializing AI engine...")
        self.lbl_loading_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_loading_title.setStyleSheet(
            "color: #2C3E50; font-size: 12pt; font-weight: 600;"
        )

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(14)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 7px;
                background-color: #E9ECEF;
                color: #495057;
                font-size: 8pt;
                text-align: center;
            }
            QProgressBar::chunk {
                border-radius: 7px;
                background-color: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4DABF7, stop:1 #228BE6
                );
            }
        """)

        self.lbl_loading_status = QLabel("Preparing...")
        self.lbl_loading_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_loading_status.setWordWrap(True)
        self.lbl_loading_status.setStyleSheet(
            "color: #6C757D; font-size: 9pt;"
        )

        loading_v.addWidget(self.lbl_loading_title)
        loading_v.addWidget(self.progress_bar)
        loading_v.addWidget(self.lbl_loading_status)
        main_vbox.addWidget(self.loading_container)
        # --- Error details (hidden until a failure occurs) ---
        self.btn_show_details = QPushButton("View technical details ▾")
        self.btn_show_details.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_show_details.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #868E96;
                border: none;
                font-size: 9pt;
                text-decoration: underline;
            }
            QPushButton:hover { color: #495057; }
        """)
        self.btn_show_details.clicked.connect(self._toggle_error_details)
        self.btn_show_details.setVisible(False)

        self.txt_error_details = QTextEdit()
        self.txt_error_details.setReadOnly(True)
        self.txt_error_details.setMaximumHeight(160)
        self.txt_error_details.setStyleSheet("""
            QTextEdit {
                background-color: #F8F9FA;
                color: #6C757D;
                border: 1px solid #E9ECEF;
                border-radius: 6px;
                font-family: 'Menlo', 'Consolas', monospace;
                font-size: 8pt;
                padding: 8px;
            }
        """)
        self.txt_error_details.setVisible(False)

        h_btn = QHBoxLayout()
        h_btn.addStretch()
        h_btn.addWidget(self.btn_show_details)
        h_btn.addStretch()
        main_vbox.addLayout(h_btn)
        main_vbox.addWidget(self.txt_error_details)

        # --- Middle: Borderless elegant chat flow (hidden until ready) ---
        self.txt_history = QTextEdit()
        self.txt_history.setReadOnly(True)
        self.txt_history.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Use system font: SF on macOS, Segoe UI on Windows. Avoids the
        # "missing font Segoe UI" warning on Mac.
        font = QFont()
        font.setFamily(".AppleSystemUIFont" if _IS_MAC else "Segoe UI")
        font.setPointSize(self.BASE_FONT_SIZE)
        self.txt_history.setFont(font)
        self.inp_font = font

        self.txt_history.setVisible(False)
        main_vbox.addWidget(self.txt_history)

        # --- Bottom: Modern input area ---
        input_layout = QHBoxLayout()
        input_layout.setSpacing(10)

        self.inp = QLineEdit()
        self.inp.setPlaceholderText("Message AI Copilot...")
        self.inp.setFont(self.inp_font)
        self.inp.returnPressed.connect(self.on_send)

        self.btn_send = QPushButton("Send")
        self.btn_send.setObjectName("SendBtn")
        self.btn_send.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_send.clicked.connect(self.on_send)
        self.btn_send.setAutoDefault(True)
        self.btn_send.setDefault(True)
        input_layout.addWidget(self.inp)
        input_layout.addWidget(self.btn_send)
        main_vbox.addLayout(input_layout)

        self.inp.setEnabled(False)
        self.btn_send.setEnabled(False)

        # Incremental stdout buffer (must exist before _start_process emits anything)
        self._stdout_buffer = ""

        # Launch subprocess
        self._start_process()

    def _set_preview_image(self, frame_rgb):
        """Helper function to set and scale the preview image smoothly."""
        h, w, c = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, c * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            220, 220, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.lbl_img.setPixmap(pixmap)

    def on_update_image(self):
        if not self.parent.thread or not self.parent.thread.isRunning():
            from PyQt6.QtWidgets import QMessageBox
            msg = QMessageBox(self)
            msg.setWindowTitle("Tracker Paused")
            msg.setText("Real-time tracking is currently stopped.\nPlease start it in the main window to grab a new frame.")
            msg.setStyleSheet("background-color: white; color: black;")
            msg.exec()
            return

        if getattr(self.parent, 'latest_frame', None) is None:
            return

        new_frame = self.parent.latest_frame.copy()

        if np.array_equal(self.frame_rgb, new_frame):
            return

        self.frame_rgb = new_frame
        self._set_preview_image(self.frame_rgb)

        Image.fromarray(new_frame).save(self.temp_img_path)

        if self.process and self.process.state() == QProcess.ProcessState.Running:
            msg = json.dumps({"type": "update_image", "path": self.temp_img_path}) + "\n"
            self.process.write(msg.encode("utf-8"))

        # cool visual feedback for image update
        sys_html = """
        <table width="100%" cellspacing="0" cellpadding="0">
          <tr>
            <td align="center">
              <table cellspacing="0" cellpadding="6" bgcolor="#F8F9FA">
                <tr>
                  <td><span style="color: #868E96; font-size: 10pt;">👁️ Vision synchronized with current microscope view</span></td>
                </tr>
              </table>
            </td>
          </tr>
        </table>
        """
        self.txt_history.append(sys_html)

    def _start_process(self):
        self.process = QProcess(self)

        # Merge stderr into stdout so any stray prints don't get lost — the
        # JSON parser already tolerates non-JSON lines.
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)

        # Default precision differs by platform: bitsandbytes 8bit needs CUDA,
        # so on Mac we ship a 16bit model instead.
        if not self.llm_precision:
            self.llm_precision = "16bit" if _IS_MAC else "8bit"

        if getattr(sys, 'frozen', False):
            self.process.setProgram(sys.executable)
            self.process.setArguments([
                "--run-llm-worker",
                "--cfg", self.llm_metadata_path,
                "--image", self.temp_img_path,
                "--precision", self.llm_precision,
            ])
        else:
            script_path = resource_path("llm_worker_process.py")
            self.process.setProgram(sys.executable)
            # -u keeps the child's stdout/stderr unbuffered so progress JSON
            # lines reach us in real time (important on Windows in particular).
            self.process.setArguments([
                "-u", script_path,
                "--cfg", self.llm_metadata_path,
                "--image", self.temp_img_path,
                "--precision", self.llm_precision,
            ])

        self.process.readyReadStandardOutput.connect(self._handle_stdout)
        self.process.finished.connect(self._on_process_finished)
        self.process.errorOccurred.connect(self._on_process_error)

        self.process.start()

    def _on_process_finished(self, exitCode, exitStatus):
            if exitCode != 0 and not self.model_ready:
                hint = ""
                if exitCode in (137, -9):
                    hint = "The OS killed the process — most likely it ran out of RAM."
                self._show_friendly_error(
                    f"Backend process exited with code {exitCode}. {hint}".strip()
                )

    def _on_process_error(self, error):
        self.lbl_loading_status.setText(f"Failed to start process: {error}")

    def _handle_stdout(self):
        # Read whatever bytes are currently available and append to buffer.
        raw_bytes = self.process.readAllStandardOutput().data()
        self._stdout_buffer += raw_bytes.decode("utf-8", errors="replace")

        # Process every complete (newline-terminated) line.
        while "\n" in self._stdout_buffer:
            line, self._stdout_buffer = self._stdout_buffer.split("\n", 1)
            line = line.strip()

            if not line:
                continue

            try:
                msg = json.loads(line)
                msg_type = msg.get("type")

                if msg_type == "progress":
                    pct = msg.get("percent", -1)
                    txt = msg.get("text", "")
                    if isinstance(pct, int) and 0 <= pct <= 100:
                        self.progress_bar.setValue(pct)
                    if txt:
                        self.lbl_loading_status.setText(txt)

                elif msg_type == "ready":
                    self.model_ready = True
                    self.progress_bar.setValue(100)
                    self.loading_container.setVisible(False)
                    self.txt_history.setVisible(True)
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
                    cursor = self.txt_history.textCursor()
                    cursor.movePosition(QTextCursor.MoveOperation.End)

                    fmt = cursor.charFormat()
                    fmt.setFontPointSize(self.BASE_FONT_SIZE)
                    fmt.setForeground(QColor("#2C3E50"))
                    cursor.setCharFormat(fmt)

                    cursor.insertText(msg["text"])
                    self.txt_history.setTextCursor(cursor)
                    self.txt_history.verticalScrollBar().setValue(
                        self.txt_history.verticalScrollBar().maximum()
                    )

                elif msg_type == "stream_end":
                    self.inp.setEnabled(True)
                    self.btn_send.setEnabled(True)
                    self.inp.setFocus()
                    self.txt_history.append("<br>")

                elif msg_type == "error":
                    if not self.model_ready:
                        self._show_friendly_error(msg.get("text", ""))
                    else:
                        # Runtime error after model loaded — show inline in chat.
                        self.txt_history.append(
                            f"<b style='color:#FA5252;'>Error:</b> {msg.get('text', '')}<br>"
                        )
                        self.inp.setEnabled(True)
                        self.btn_send.setEnabled(True)

            except json.JSONDecodeError:
                # Non-JSON line: probably stderr debug output from a library.
                # Don't surface it to the user but keep it for the dev console.
                print(f"[LLM Debug]: {line}")
            except Exception as e:
                print(f"[UI Update Error]: {e}")
    def _show_friendly_error(self, raw_text):
        """Replace the loading UI with a clean error card."""
        if self._error_shown:
            return
        self._error_shown = True

        title, message, suggestion = self._classify_error(raw_text or "")

        # Hide the progress bar — there's nothing more to load.
        self.progress_bar.setVisible(False)

        self.lbl_loading_title.setText(title)
        self.lbl_loading_title.setStyleSheet(
            "color: #E03131; font-size: 13pt; font-weight: 600;"
        )

        # Use rich text for a small visual hierarchy: message in dark, suggestion in lighter grey.
        self.lbl_loading_status.setText(
            f"<div style='color:#343A40; font-size:10pt;'>{message}</div>"
            f"<div style='color:#868E96; font-size:9pt; margin-top:8px;'>{suggestion}</div>"
        )
        self.lbl_loading_status.setTextFormat(Qt.TextFormat.RichText)

        # Stash the raw traceback in the collapsible details panel.
        if raw_text:
            self.txt_error_details.setPlainText(raw_text)
            self.btn_show_details.setVisible(True)

        # Make sure the chat area stays hidden — we never got to chat.
        self.txt_history.setVisible(False)
        self.loading_container.setVisible(True)

    def _toggle_error_details(self):
        visible = self.txt_error_details.isVisible()
        self.txt_error_details.setVisible(not visible)
        self.btn_show_details.setText(
            "Hide technical details ▴" if not visible else "View technical details ▾"
        )

    def _classify_error(self, raw_text):
        """Map a raw traceback to (title, message, suggestion) tuple."""
        t = raw_text.lower()

        if "mps backend out of memory" in t or "cuda out of memory" in t \
           or "out of memory" in t or "killed" in t:
            return (
                "💾 Not Enough Memory",
                "This Mac doesn't have enough RAM to load the AI model.",
                "Qwen2.5-VL needs about 16 GB of unified memory to run. "
                "Please try OnSight on a machine with more RAM, "
                "or contact the developer for a lighter model variant."
            )

        if "no space left" in t or "errno 28" in t or "disk" in t and "full" in t:
            return (
                "💿 Disk Full",
                "There isn't enough free space to download or load the model.",
                "Free up at least 10 GB on your startup disk, then try again."
            )

        if any(k in t for k in ("connectionerror", "httperror", "max retries",
                                 "name resolution", "name or service not known",
                                 "failed to establish")):
            return (
                "🌐 Network Issue",
                "Could not reach the model server to download weights.",
                "Check your internet connection and try again. "
                "If you're behind a corporate proxy or VPN, the Hugging Face servers may be blocked."
            )

        if "permission denied" in t or "errno 13" in t:
            return (
                "🔒 Permission Denied",
                "OnSight can't read or write the model cache folder.",
                "Check folder permissions in:\n"
                "~/Library/Application Support/OnSightPathology/"
            )

        if "no module named" in t or "modulenotfounderror" in t:
            return (
                "📦 Missing Dependency",
                "A required Python package isn't installed in this environment.",
                "Try reinstalling OnSight, or check the README for setup instructions."
            )

        if "huggingfacehub" in t or "repository not found" in t or "401" in t or "403" in t:
            return (
                "🤖 Model Access Issue",
                "The AI model could not be retrieved from Hugging Face.",
                "The model may be private or require a token. "
                "Check your Hugging Face login or the model config file."
            )

        # Generic fallback
        return (
            "⚠️ Failed to Start AI Engine",
            "Something went wrong while initializing the language model.",
            "Tap 'View technical details' below to see the full error, "
            "then send it to the developer if the problem persists."
        )
    def on_send(self):
            text = self.inp.text().strip()
            if not self.model_ready or not text:
                return

            self.inp.clear()

            # 1. User's blue bubble
            user_html = f"""
            <table width="100%" cellspacing="0" cellpadding="0">
            <tr>
                <td width="20%"></td>
                <td align="right">
                <table cellspacing="0" cellpadding="10" bgcolor="#339AF0">
                    <tr>
                    <td><span style="color: white; font-size: 12pt;">{text}</span></td>
                    </tr>
                </table>
                </td>
            </tr>
            </table>
            """
            self.txt_history.append(user_html)

            # 2. prefix
            ai_start_html = """
            <table width="100%" cellspacing="0" cellpadding="0">
            <tr>
                <td><b style="color: #20C997; font-size: 12pt;">✦ Assistant</b></td>
            </tr>
            </table>
            """
            self.txt_history.append(ai_start_html)

            self.inp.setEnabled(False)
            self.btn_send.setEnabled(False)

            if self.process:
                msg = json.dumps({"type": "prompt", "text": text}) + "\n"
                self.process.write(msg.encode("utf-8"))
                self.process.waitForBytesWritten()

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