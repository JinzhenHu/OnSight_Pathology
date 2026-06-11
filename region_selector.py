"""
region_selector.py — Non-blocking mouse capture for region selection.
"""

from PyQt6.QtCore import QObject, QThread, pyqtSignal


class _TwoClickWorker(QObject):
    """Listens for TWO mouse clicks, emits after each, then exits.
    
    Single Listener / single CGEventTap — required for macOS reliability.
    """
    first_clicked = pyqtSignal(int, int)
    both_clicked = pyqtSignal(int, int, int, int)

    def run(self):
        from pynput import mouse
        clicks = []

        def _on_click(x, y, button, pressed):
            if not pressed:
                return
            clicks.append((int(x), int(y)))
            if len(clicks) == 1:
                self.first_clicked.emit(clicks[0][0], clicks[0][1])
            elif len(clicks) >= 2:
                return False  

        with mouse.Listener(on_click=_on_click) as listener:
            listener.join()

        if len(clicks) >= 2:
            x1, y1 = clicks[0]
            x2, y2 = clicks[1]
            self.both_clicked.emit(x1, y1, x2, y2)


def capture_two_clicks(on_complete, on_first_done=None):
    """Capture two mouse clicks without blocking the main UI thread."""
    state = {"thread": None, "worker": None}

    thread = QThread()
    worker = _TwoClickWorker()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)

    if on_first_done:
        worker.first_clicked.connect(lambda x, y: on_first_done())

    def _done(x1, y1, x2, y2):
        thread.quit()
        thread.wait(500)
        on_complete(x1, y1, x2, y2)

    worker.both_clicked.connect(_done)
    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread.start()

    state["thread"] = thread
    state["worker"] = worker
    return state