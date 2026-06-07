import logging
import os
import sys
from PyQt6.QtWidgets import QMessageBox
from logging.handlers import RotatingFileHandler
import platform
from datetime import datetime


def get_user_log_path(filename="app.log"):
    local_appdata = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    log_dir = os.path.join(local_appdata, "OnSightPathology", "Logs")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, filename)

log_path = get_user_log_path()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        #RotatingFileHandler(log_path, mode='a', maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'),
        logging.FileHandler(log_path, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
# Silence noisy third-party loggers
for noisy in [
    "PIL",
    "PIL.PngImagePlugin",
    "PIL.Image",
    "matplotlib",
    "matplotlib.font_manager",
    "matplotlib.pyplot",
    "numexpr",
    "numexpr.utils",
    "urllib3",
    "filelock",
    "huggingface_hub",
    "transformers",
    "timm",
    "fsspec",
    "asyncio",
]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

logging.info("=" * 70)
logging.info(f"App started at {datetime.now().isoformat()}")
logging.info(f"Python: {sys.version.split()[0]}")
logging.info(f"Platform: {platform.platform()}")
logging.info(f"Build type: {os.environ.get('BUILD_TYPE', 'GPU')}")
try:
    import torch
    logging.info(f"PyTorch: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logging.info(f"GPU: {props.name}, VRAM: {props.total_memory/1e9:.2f} GB")
except Exception as e:
    logging.warning(f"Torch info unavailable: {e}")
logging.info("=" * 70)





##############################################################################################
##############################################################################################
# ============================================================
# Developer contact info
# ============================================================
DEVELOPER_EMAIL = "jinzhen.hu@mail.utoronto.ca"


def _open_log_folder():
    """Open the folder containing app.log in the OS file manager."""
    import subprocess
    log_dir = os.path.dirname(log_path)
    try:
        if sys.platform == "win32":
            os.startfile(log_dir)
        elif sys.platform == "darwin":
            subprocess.run(["open", log_dir])
        else:
            subprocess.run(["xdg-open", log_dir])
    except Exception as e:
        logging.warning(f"Could not open log folder: {e}")


def _compose_error_email(exc_type, exc_value, exc_traceback):
    """
    Attempt to open mail client AND copy report to clipboard AND save to file.
    The user can use whichever path works on their system.
    """
    import urllib.parse
    import traceback
    import webbrowser
    import platform
    from datetime import datetime

    tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))

    # ---- Collect system info ----
    try:
        import torch
        gpu_info = "N/A"
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu_info = f"{props.name} ({props.total_memory/1e9:.1f} GB VRAM)"
        torch_info = f"{torch.__version__}, CUDA: {torch.cuda.is_available()}"
    except Exception:
        gpu_info = "unavailable"
        torch_info = "unavailable"

    # ---- Build full report ----
    subject = f"[OnSight Bug Report] {exc_type.__name__}"
    body = f"""Hi Jinzhen,

I encountered an unexpected error while using OnSight Pathology.

----- System Info -----
OS:        {platform.platform()}
Python:    {sys.version.split()[0]}
PyTorch:   {torch_info}
GPU:       {gpu_info}
Build:     {os.environ.get('BUILD_TYPE', 'GPU')}
Time:      {datetime.now().isoformat()}

----- Error -----
{exc_type.__name__}: {exc_value}

----- Traceback -----
{tb_str}

----- What I was doing when this happened -----
(Please describe briefly what you clicked / which model you selected, etc.)


Thanks!
"""

    full_text_for_clipboard = (
        f"To: {DEVELOPER_EMAIL}\n"
        f"Subject: {subject}\n\n"
        f"{body}"
    )

    # ---- Path 1: copy to clipboard (most reliable) ----
    try:
        from PyQt6.QtWidgets import QApplication
        QApplication.clipboard().setText(full_text_for_clipboard)
        clipboard_ok = True
    except Exception as e:
        logging.warning(f"Clipboard copy failed: {e}")
        clipboard_ok = False

    # ---- Path 2: save report to a file next to the log ----
    report_path = None
    try:
        log_dir = os.path.dirname(log_path)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(log_dir, f"bug_report_{stamp}.txt")
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(full_text_for_clipboard)
    except Exception as e:
        logging.warning(f"Could not write bug report file: {e}")
        report_path = None

    # ---- Path 3: try to launch mail client ----
    mailto_ok = False
    try:
        encoded_body = urllib.parse.quote(body)
        encoded_subject = urllib.parse.quote(subject)
        url = f"mailto:{DEVELOPER_EMAIL}?subject={encoded_subject}&body={encoded_body}"
        if len(url) > 2000:
            short_body = (
                f"Hi Jinzhen,\n\n"
                f"I encountered an unexpected error in OnSight Pathology:\n\n"
                f"{exc_type.__name__}: {str(exc_value)[:200]}\n\n"
                f"The full log file is attached (please attach app.log manually from the log folder).\n\n"
                f"Thanks!"
            )
            url = (f"mailto:{DEVELOPER_EMAIL}"
                   f"?subject={encoded_subject}"
                   f"&body={urllib.parse.quote(short_body)}")
        mailto_ok = webbrowser.open(url)
    except Exception as e:
        logging.warning(f"Could not open mail client: {e}")

    # ---- Tell the user what happened ----
    from PyQt6.QtWidgets import QMessageBox
    
    lines = ["<b>Bug report prepared.</b><br>"]
    
    if mailto_ok:
        lines.append("✅ Your email client should open shortly with the report pre-filled.<br>")
    else:
        lines.append("⚠️ Could not open your default email client.<br>")
    
    if clipboard_ok:
        lines.append("✅ The full report has been <b>copied to your clipboard</b> — "
                     "you can paste it into Gmail, Outlook Web, or any email service.<br>")
    
    if report_path:
        lines.append(f"✅ A copy of the report was saved to:<br><code>{report_path}</code><br>")
    
    lines.append(f"<br>Please send to: <b>{DEVELOPER_EMAIL}</b>")

    info = QMessageBox()
    info.setIcon(QMessageBox.Icon.Information)
    info.setWindowTitle("Bug Report Ready")
    info.setTextFormat(Qt.TextFormat.RichText if False else 1)  # RichText
    info.setText("".join(lines))
    
    btn_open_folder = info.addButton("📂 Open Log Folder", QMessageBox.ButtonRole.ActionRole)
    info.addButton("OK", QMessageBox.ButtonRole.AcceptRole)
    info.exec()
    
    if info.clickedButton() is btn_open_folder:
        _open_log_folder()


# ============================================================
# Global exception hook
# ============================================================
def log_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    from PyQt6.QtWidgets import (
        QApplication, QDialog, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QTextEdit, QDialogButtonBox
    )
    from PyQt6.QtCore import Qt
    app = QApplication.instance()
    if not app:
        return

    import traceback as _tb
    full_tb = "".join(_tb.format_exception(exc_type, exc_value, exc_traceback))

    dlg = QDialog()
    dlg.setWindowTitle("Unexpected Error")
    dlg.setMinimumWidth(520)

    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(16, 16, 16, 16)
    layout.setSpacing(10)

    # --- Header ---
    header = QLabel("<b>OnSight Pathology encountered an unexpected error.</b>")
    header.setWordWrap(True)
    layout.addWidget(header)

    # --- Short error message ---
    short = QLabel(f"<b>{exc_type.__name__}:</b> {str(exc_value)[:300]}")
    short.setWordWrap(True)
    short.setStyleSheet("color: #c0392b;")
    layout.addWidget(short)

    # --- Hint ---
    hint = QLabel(
        "<small>A diagnostic log has been saved. "
        "You can email the developer, open the log folder, or copy details below.</small>"
    )
    hint.setWordWrap(True)
    layout.addWidget(hint)

    # --- Collapsible traceback ---
    tb_view = QTextEdit()
    tb_view.setReadOnly(True)
    tb_view.setPlainText(full_tb)
    tb_view.setStyleSheet("font-family: Consolas, monospace; font-size: 9pt;")
    tb_view.setVisible(False)
    layout.addWidget(tb_view)

    btn_toggle = QPushButton("Show Details ▼")
    def _toggle_details():
        is_vis = tb_view.isVisible()
        tb_view.setVisible(not is_vis)
        btn_toggle.setText("Hide Details ▲" if not is_vis else "Show Details ▼")
        dlg.adjustSize()
    btn_toggle.clicked.connect(_toggle_details)
    layout.addWidget(btn_toggle)

    # --- Action buttons (non-closing) ---
    actions = QHBoxLayout()

    btn_email = QPushButton("📧 Email Developer")
    btn_email.clicked.connect(
        lambda: _compose_error_email(exc_type, exc_value, exc_traceback)
    )
    actions.addWidget(btn_email)

    btn_open = QPushButton("📂 Open Log Folder")
    btn_open.clicked.connect(_open_log_folder)
    actions.addWidget(btn_open)

    btn_copy = QPushButton("📋 Copy Log Path")
    def _copy_path():
        QApplication.clipboard().setText(log_path)
        # Brief visual feedback
        btn_copy.setText("✓ Copied!")
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(1500, lambda: btn_copy.setText("📋 Copy Log Path"))
    btn_copy.clicked.connect(_copy_path)
    actions.addWidget(btn_copy)

    layout.addLayout(actions)

    # --- Close button (the ONLY one that closes the dialog) ---
    close_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    close_box.rejected.connect(dlg.reject)
    layout.addWidget(close_box)

    dlg.exec()


sys.excepthook = log_exception



##############################################################################################
##############################################################################################
# Threading hook (Python threads) 
import threading

def thread_excepthook(args):
    if issubclass(args.exc_type, KeyboardInterrupt):
        return
    logging.critical(
        f"Unhandled exception in thread '{args.thread.name}'",
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback)
    )

threading.excepthook = thread_excepthook


#  Qt message handler (C++ side) ----------
try:
    from PyQt6.QtCore import qInstallMessageHandler, QtMsgType

    _qt_level_map = {
        QtMsgType.QtDebugMsg: logging.DEBUG,
        QtMsgType.QtInfoMsg: logging.INFO,
        QtMsgType.QtWarningMsg: logging.WARNING,
        QtMsgType.QtCriticalMsg: logging.ERROR,
        QtMsgType.QtFatalMsg: logging.CRITICAL,
    }

    def qt_message_handler(mode, context, message):
        logging.log(_qt_level_map.get(mode, logging.INFO), f"[Qt] {message}")

    qInstallMessageHandler(qt_message_handler)
except Exception as e:
    logging.warning(f"Qt message handler not installed: {e}")