"""
region_selector.py — Non-blocking mouse capture for region selection.

"""

from PyQt6.QtCore import QObject, QThread, pyqtSignal


class _MouseClickWorker(QObject):
    """Listens for ONE mouse click, emits its (x, y), then exits."""
    clicked = pyqtSignal(int, int)

    def run(self):
        from pynput import mouse
        result = []

        def _on_click(x, y, button, pressed):
            if pressed:
                result.append((int(x), int(y)))
                return False  # stops the Listener

        with mouse.Listener(on_click=_on_click) as listener:
            listener.join()

        if result:
            self.clicked.emit(result[0][0], result[0][1])
        else:
            self.clicked.emit(0, 0)


def capture_two_clicks(on_complete, on_first_done=None):
    """Capture two mouse clicks without blocking the main UI thread.
    Returns the QThread holding worker so caller can keep a reference.
    """
    state = {"x1": None, "y1": None, "threads": []}

    def _start_second():
        thread2 = QThread()
        worker2 = _MouseClickWorker()
        worker2.moveToThread(thread2)
        thread2.started.connect(worker2.run)

        def _second_click(x2, y2):
            thread2.quit()
            thread2.wait(500)
            on_complete(state["x1"], state["y1"], x2, y2)

        worker2.clicked.connect(_second_click)
        thread2.finished.connect(worker2.deleteLater)
        thread2.finished.connect(thread2.deleteLater)
        thread2.start()
        state["threads"].append((thread2, worker2))

    def _first_click(x, y):
        state["x1"] = x
        state["y1"] = y
        thread1.quit()
        if on_first_done:
            on_first_done()
        _start_second()

    thread1 = QThread()
    worker1 = _MouseClickWorker()
    worker1.moveToThread(thread1)
    thread1.started.connect(worker1.run)
    worker1.clicked.connect(_first_click)
    thread1.finished.connect(worker1.deleteLater)
    thread1.finished.connect(thread1.deleteLater)
    thread1.start()
    state["threads"].append((thread1, worker1))
    return state