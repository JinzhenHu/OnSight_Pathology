import sys
import time
import json
import numpy as np
import gc
import os

import torch
from PyQt6.QtWidgets import (
    QVBoxLayout, QPushButton, QLabel, QLineEdit, QDialog, QTextEdit
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtGui import QTextCursor
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QProcess
from PIL import Image
import tempfile


class LLMChatDialog(QDialog):
    """One-off chat about the latest inference frame."""

    def __init__(self, frame_rgb: np.ndarray, llm_metadata_path, llm_precision, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Chat with LLM")
        self.resize(480, 600)

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

        # ---------------------- GUI  -----------------------

        # Preview image
        vbox = QVBoxLayout(self)
        lbl_img = QLabel()
        lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        h, w, c = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, c * w, QImage.Format.Format_RGB888)
        lbl_img.setPixmap(QPixmap.fromImage(qimg).scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio))
        vbox.addWidget(lbl_img)

        self.txt_history = QTextEdit()
        self.txt_history.setReadOnly(True)
        vbox.addWidget(self.txt_history)

        self.inp = QLineEdit()
        self.inp.setPlaceholderText("Ask something about the image …")
        vbox.addWidget(self.inp)

        btn_send = QPushButton("Send")
        btn_send.clicked.connect(self.on_send)
        vbox.addWidget(btn_send)

        self.txt_history.setText("<b>Note: Loading the LLM for the first time can take a long time</b>")
        self.txt_history.append("<i>Please wait, loading model…</i>")
        self.inp.setEnabled(False)

        # Launch subprocess to handle model
        self._start_process()

    # In LLMChatDialog._start_process
    def _start_process(self):
        self.process = QProcess(self)

        if not self.llm_precision:
            self.llm_precision = "8bit"

        if getattr(sys, 'frozen', False):
            # PyInstaller bundled executable — use .exe
            script_path = os.path.join(os.path.dirname(sys.executable), "llm_worker_process.exe")
            self.process.setProgram(script_path)
            self.process.setArguments([
                "--cfg", self.llm_metadata_path,
                "--image", self.temp_img_path,
                "--precision", self.llm_precision,
            ])
        else:
            # Running locally — use .py
            script_path = os.path.join(os.path.dirname(__file__), "llm_worker_process.py")
            self.process.setProgram(sys.executable)
            self.process.setArguments([
                script_path,
                "--cfg", self.llm_metadata_path,
                "--image", self.temp_img_path,
                "--precision", self.llm_precision,
            ])

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
                    self.txt_history.append("<i>Model loaded. You can start chatting.</i>")
                    self.inp.setEnabled(True)
                    self.inp.setFocus()

                elif msg_type == "chunk":
                    # a piece of the stream, append it without a newline
                    self.txt_history.moveCursor(QTextCursor.MoveOperation.End)
                    self.txt_history.insertPlainText(msg["text"])

                elif msg_type == "stream_end":
                    # The stream is finished, re-enable input
                    self.inp.setEnabled(True)
                    self.inp.setFocus()

                elif msg_type == "error":
                    self._append("Error", msg["text"])
                    self.inp.setEnabled(True)
                    self.inp.setFocus()

            except Exception as e:
                print(f"Malformed JSON or error processing message: {line}, Error: {e}")

    def closeEvent(self, event):
        if self.process and self.process.state() == QProcess.ProcessState.Running:
            self.process.kill()
            self.process.finished.connect(self._cleanup_temp_file)
            self.process.waitForFinished()  # Ensure process exits
        else:
            self._cleanup_temp_file()

        super().closeEvent(event)

    def _cleanup_temp_file(self):
        if os.path.exists(self.temp_img_path):
            try:
                os.remove(self.temp_img_path)
            except Exception as e:
                print(f"Could not delete temp image: {e}")

    def on_send(self):
        text = self.inp.text().strip()
        if not self.model_ready or not text:
            return

        self.inp.clear()
        self._append("User", text)

        # Prepare the assistant's response area immediately
        self.txt_history.append("<b>Assistant:</b> ")
        self.inp.setEnabled(False)

        if self.process:
            msg = json.dumps({"type": "prompt", "text": text}) + "\n"
            self.process.write(msg.encode("utf-8"))

    def _append(self, who, txt):
        self.txt_history.append(f"<b>{who}:</b> {txt}")

