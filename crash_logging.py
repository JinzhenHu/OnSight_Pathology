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
# Global exception hook
def log_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    from PyQt6.QtWidgets import QApplication, QMessageBox, QPushButton
    app = QApplication.instance()
    if not app:
        return

    msg = QMessageBox()
    msg.setIcon(QMessageBox.Icon.Critical)
    msg.setWindowTitle("Unexpected Error")
    msg.setText("<b>OnSight Pathology encountered an unexpected error.</b>")
    msg.setInformativeText(
        #f"<p>{exc_type.__name__}: {str(exc_value)[:300]}</p>"
        f"<p>A diagnostic log has been saved. "
        f"Please share it with the developer to help fix this issue.</p>"
    )
    msg.setDetailedText("".join(__import__("traceback").format_exception(exc_type, exc_value, exc_traceback)))

    btn_open = msg.addButton("📂 Open Log Folder", QMessageBox.ButtonRole.ActionRole)
    btn_copy = msg.addButton("📋 Copy Log Path", QMessageBox.ButtonRole.ActionRole)
    btn_close = msg.addButton("Close", QMessageBox.ButtonRole.RejectRole)
    msg.setDefaultButton(btn_open)

    msg.exec()

    clicked = msg.clickedButton()
    if clicked is btn_open:
        _open_log_folder()
    elif clicked is btn_copy:
        QApplication.clipboard().setText(log_path)


def _open_log_folder():
    log_dir = os.path.dirname(log_path)
    try:
        if sys.platform == "win32":
            os.startfile(log_dir)
        elif sys.platform == "darwin":
            __import__("subprocess").run(["open", log_dir])
        else:
            __import__("subprocess").run(["xdg-open", log_dir])
    except Exception as e:
        logging.warning(f"Could not open log folder: {e}")

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