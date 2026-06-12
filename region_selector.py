"""
region_selector.py — Non-blocking mouse capture for region selection.

macOS-native via NSEvent.addGlobalMonitorForEventsMatchingMask_handler_,
which avoids the CGEventTap 1-second callback timeout that breaks
pynput in packaged .app bundles.

Falls back to pynput on Windows/Linux where CGEventTap is irrelevant.
"""

import sys
from PyQt6.QtCore import QObject, QThread, pyqtSignal, QTimer


# ============================================================================
# macOS path: NSEvent global monitor (no CGEventTap timeout issues)
# ============================================================================

class _MacClickCapture(QObject):
    """Uses NSEvent.addGlobalMonitorForEventsMatchingMask_handler_.
    
    Lives on the main thread. Non-blocking. No QThread needed.
    Handler is dispatched by Cocoa's main runloop — same loop Qt uses.
    """
    first_clicked = pyqtSignal(int, int)
    both_clicked = pyqtSignal(int, int, int, int)

    def __init__(self):
        super().__init__()
        self._clicks = []
        self._monitor_global = None
        self._monitor_local = None  # for clicks on our own window

    def start(self):
        try:
            from AppKit import NSEvent, NSEventMaskLeftMouseDown
            from Quartz import CGEventGetLocation
        except ImportError as e:
            raise RuntimeError(
                f"pyobjc not available — required for macOS region selection: {e}"
            )

        # Global monitor: clicks OUTSIDE our app (microscope, slide viewer, etc.)
        self._monitor_global = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
            NSEventMaskLeftMouseDown,
            self._handle_event,
        )
        # Local monitor: clicks ON our own window (NSEvent global monitor doesn't
        # see these, so we need a separate one). Local monitor handler must
        # return the event (to let it propagate) or nil (to swallow it).
        self._monitor_local = NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
            NSEventMaskLeftMouseDown,
            self._handle_event_local,
        )

    def _handle_event_local(self, ns_event):
        self._handle_event(ns_event)
        return ns_event  # let Qt also receive it normally

    def _handle_event(self, ns_event):
        # NSEvent uses bottom-left origin; convert to top-left (Qt / mss convention)
        from AppKit import NSScreen
        loc = ns_event.locationInWindow()
        # For global monitor there's no window context; use mouseLocation instead
        from AppKit import NSEvent as _NSE
        mouse_loc = _NSE.mouseLocation()
        # Flip Y: NSScreen's main screen height minus y
        main_screen = NSScreen.mainScreen()
        screen_h = main_screen.frame().size.height
        x = int(mouse_loc.x)
        y = int(screen_h - mouse_loc.y)

        self._clicks.append((x, y))
        if len(self._clicks) == 1:
            self.first_clicked.emit(x, y)
        elif len(self._clicks) >= 2:
            x1, y1 = self._clicks[0]
            x2, y2 = self._clicks[1]
            self.stop()
            self.both_clicked.emit(x1, y1, x2, y2)

    def stop(self):
        from AppKit import NSEvent
        if self._monitor_global is not None:
            NSEvent.removeMonitor_(self._monitor_global)
            self._monitor_global = None
        if self._monitor_local is not None:
            NSEvent.removeMonitor_(self._monitor_local)
            self._monitor_local = None


# ============================================================================
# Windows / Linux path: keep pynput (works fine outside macOS)
# ============================================================================

class _PynputClickWorker(QObject):
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


# ============================================================================
# Public API — same signature as before
# ============================================================================

def capture_two_clicks(on_complete, on_first_done=None):
    """Capture two mouse clicks without blocking the main UI thread."""
    state = {}

    if sys.platform == "darwin":
        # macOS — NSEvent monitor on main thread
        capture = _MacClickCapture()

        if on_first_done:
            capture.first_clicked.connect(lambda x, y: on_first_done())

        def _done(x1, y1, x2, y2):
            on_complete(x1, y1, x2, y2)

        capture.both_clicked.connect(_done)
        capture.start()
        state["capture"] = capture  # keep alive
        return state

    else:
        # Windows / Linux — pynput on QThread
        thread = QThread()
        worker = _PynputClickWorker()
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