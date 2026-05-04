import logging
import os
import sys
from PyQt6.QtWidgets import QMessageBox

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
        logging.FileHandler(log_path, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.info("App started")

# Global exception hook
def log_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app:
        QMessageBox.critical(None, "Error", f"An unexpected error occurred.\nLog saved to:\n{log_path}")

sys.excepthook = log_exception