import logging
import os
import sys
import platform
import threading
from datetime import datetime
from logging.handlers import RotatingFileHandler

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox


# ============================================================
# Log file setup
# ============================================================
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
        # RotatingFileHandler(log_path, mode='a', maxBytes=5*1024*1024,
        #                     backupCount=3, encoding='utf-8'),
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

# Startup banner
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


# ============================================================
# Email composer
# ============================================================
def _compose_error_email(exc_type, exc_value, exc_traceback, tb_str=None):
    """
    Build a bug report and try multiple paths to deliver it:
      1. Copy to clipboard
      2. Save to file in log folder
      3. Open mail client via mailto:

    Only shows a fallback dialog if the mail client could NOT be opened.
    """
    try:
        import urllib.parse
        import traceback
        import webbrowser

        # Build traceback string
        if tb_str is None:
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

        # ---- Build subject/body ----
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

        # ---- Path 1: copy to clipboard ----
        clipboard_ok = False
        try:
            from PyQt6.QtWidgets import QApplication
            QApplication.clipboard().setText(full_text_for_clipboard)
            clipboard_ok = True
        except Exception as e:
            logging.warning(f"Clipboard copy failed: {e}")

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

        # ---- If mail client opened OK, no further dialog needed ----
        if mailto_ok:
            return

        # ---- Otherwise show fallback dialog telling user how to send manually ----
        lines = ["<b>Could not open your default email client.</b><br><br>"]

        if clipboard_ok:
            lines.append("✅ The full report has been <b>copied to your clipboard</b>. "
                         "Paste it into Gmail, Outlook Web, or any email service.<br>")

        if report_path:
            lines.append(f"✅ A copy of the report was saved to:<br>"
                         f"<code>{report_path}</code><br>")

        lines.append(f"<br>Please send to: <b>{DEVELOPER_EMAIL}</b>")

        info = QMessageBox()
        info.setIcon(QMessageBox.Icon.Warning)
        info.setWindowTitle("Manual Send Required")
        info.setTextFormat(Qt.TextFormat.RichText)
        info.setText("".join(lines))

        btn_open_folder = info.addButton("📂 Open Log Folder", QMessageBox.ButtonRole.ActionRole)
        info.addButton("OK", QMessageBox.ButtonRole.AcceptRole)
        info.exec()

        if info.clickedButton() is btn_open_folder:
            _open_log_folder()

    except Exception as e:
        # Last-resort: even the email composer itself failed.
        # Don't recurse — just log and show a basic message if possible.
        logging.error(f"Email composer failed: {e}", exc_info=True)
        try:
            QMessageBox.warning(
                None,
                "Email Helper Failed",
                f"Could not prepare email. Please manually send the log file from:\n"
                f"{os.path.dirname(log_path)}\n\nTo: {DEVELOPER_EMAIL}"
            )
        except Exception:
            pass  # nothing more we can do


# ============================================================
# Global uncaught-exception hook
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

    # --- Hint (normal-sized) ---
    hint = QLabel(
        "A diagnostic log has been saved. "
        "You can email the developer, open the log folder, or copy details below."
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
        btn_copy.setText("✓ Copied!")
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(1500, lambda: btn_copy.setText("📋 Copy Log Path"))
    btn_copy.clicked.connect(_copy_path)
    actions.addWidget(btn_copy)

    layout.addLayout(actions)

    # --- Close button ---
    close_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    close_box.rejected.connect(dlg.reject)
    layout.addWidget(close_box)

    dlg.exec()


sys.excepthook = log_exception


# ============================================================
# Reusable error dialog (for caught exceptions from app code)
# ============================================================
def show_error_dialog(title: str, short_msg: str, details: str = "",
                      hint_html: str = "", parent=None):
    """
    Show the same rich error dialog used by sys.excepthook, but for errors
    caught by application code (not uncaught exceptions).

    Args:
        title: window title, e.g. "Could not load model"
        short_msg: short error line (used in email body, NOT shown in UI)
        details: full traceback (shown under 'Show Details')
        hint_html: extra rich-text hint shown above the footnote
        parent: optional parent widget
    """
    from PyQt6.QtWidgets import (
        QApplication, QDialog, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QTextEdit, QDialogButtonBox
    )

    app = QApplication.instance()
    if not app:
        return

    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.setMinimumWidth(520)

    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(16, 16, 16, 16)
    layout.setSpacing(10)

    header = QLabel(f"<b>{title}</b>")
    header.setWordWrap(True)
    layout.addWidget(header)

    # (red short_msg removed by user request — still used for the email body)

    if hint_html:
        hint = QLabel(hint_html)
        hint.setWordWrap(True)
        hint.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(hint)

    footnote = QLabel(
        "A diagnostic log has been saved. "
        "You can email the developer or open the log folder below."
    )
    footnote.setWordWrap(True)
    footnote.setTextFormat(Qt.TextFormat.RichText)
    layout.addWidget(footnote)

    # Collapsible traceback
    if details:
        tb_view = QTextEdit()
        tb_view.setReadOnly(True)
        tb_view.setPlainText(details)
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

    # Action buttons (non-closing)
    actions = QHBoxLayout()

    btn_email = QPushButton("📧 Email Developer")
    def _send_email():
        # Build a pseudo exception class with a meaningful name for the subject
        safe_title = title.replace(" ", "_").replace("'", "")[:50] or "AppError"
        DynExc = type(safe_title, (Exception,), {})
        exc = DynExc(short_msg)
        _compose_error_email(DynExc, exc, None, tb_str=details)
    btn_email.clicked.connect(_send_email)
    actions.addWidget(btn_email)

    btn_open = QPushButton("📂 Open Log Folder")
    btn_open.clicked.connect(_open_log_folder)
    actions.addWidget(btn_open)

    btn_copy = QPushButton("📋 Copy Log Path")
    def _copy_path():
        QApplication.clipboard().setText(log_path)
        btn_copy.setText("✓ Copied!")
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(1500, lambda: btn_copy.setText("📋 Copy Log Path"))
    btn_copy.clicked.connect(_copy_path)
    actions.addWidget(btn_copy)

    layout.addLayout(actions)

    close_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    close_box.rejected.connect(dlg.reject)
    layout.addWidget(close_box)

    dlg.exec()


# ============================================================
# Threading hook (Python sub-threads)
# ============================================================
def thread_excepthook(args):
    if issubclass(args.exc_type, KeyboardInterrupt):
        return
    logging.critical(
        f"Unhandled exception in thread '{args.thread.name}'",
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback)
    )


threading.excepthook = thread_excepthook


# ============================================================
# Qt message handler (C++ side warnings/errors)
# ============================================================
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