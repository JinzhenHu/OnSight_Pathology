"""
First-run dialog that asks Windows users to set display scaling to 100%.
Plays an animated demonstration of opening a dropdown and selecting 100%,
then offers a button to open the real Windows Display Settings.
"""
import os
import sys
import json
import logging

from PyQt6.QtCore import Qt, QTimer, QRect, QPointF, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt6.QtGui import (
    QGuiApplication, QPainter, QColor, QPen, QBrush, QFont, QPolygon, QPainterPath
)
from PyQt6.QtCore import QPoint
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QFrame, QWidget, QSizePolicy, QMessageBox,
)


# ---------------------------------------------------------------------------
# Settings I/O
# ---------------------------------------------------------------------------
def _settings_path() -> str:
    local_appdata = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    settings_dir = os.path.join(local_appdata, "OnSightPathology", "Settings")
    os.makedirs(settings_dir, exist_ok=True)
    return os.path.join(settings_dir, "settings.json")


def _load_settings() -> dict:
    try:
        with open(_settings_path(), "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_settings(data: dict) -> None:
    try:
        with open(_settings_path(), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        logging.warning(f"Could not save DPI warning preference: {e}")


# ---------------------------------------------------------------------------
# Real DPI detection (Win32-backed, immune to QT_SCALE_FACTOR)
# ---------------------------------------------------------------------------
def _current_dpi_scale() -> float:
    if sys.platform == "win32":
        try:
            import ctypes
            user32 = ctypes.windll.user32
            try:
                dpi = user32.GetDpiForSystem()
                return dpi / 96.0
            except (AttributeError, OSError):
                gdi32 = ctypes.windll.gdi32
                LOGPIXELSX = 88
                hdc = user32.GetDC(0)
                dpi = gdi32.GetDeviceCaps(hdc, LOGPIXELSX)
                user32.ReleaseDC(0, hdc)
                return dpi / 96.0 if dpi else 1.0
        except Exception as e:
            logging.warning(f"Could not query Windows DPI: {e}")
            return 1.0
    else:
        try:
            primary = QGuiApplication.primaryScreen()
            return float(primary.devicePixelRatio()) if primary else 1.0
        except Exception:
            return 1.0


# ============================================================================
# AnimatedDpiDemo — the actual animated widget
# ============================================================================
class AnimatedDpiDemo(QWidget):
    """
    Self-contained looping animation that demonstrates:
      1. Hovering a Scale dropdown
      2. Clicking to open it
      3. Moving the cursor to '100%'
      4. Selecting it; dropdown closes showing 100% ✓
    """

    # Frame durations (ms)
    DURATIONS = {
        "idle":           1200,   # initial state, cursor off to the side
        "move_to_combo":   800,   # cursor glides onto the dropdown
        "open_combo":      400,   # click, dropdown pops open
        "move_to_100":     900,   # cursor glides down to 100% row
        "select_100":      500,   # row highlights, click
        "show_result":    1800,   # dropdown closed, big green checkmark
    }

    # Layout of the mock Settings panel inside this widget
    PANEL_RECT = QRect(40, 50, 320, 90)        # the white "Scale and layout" card
    COMBO_RECT = QRect(60, 92, 280, 36)        # the dropdown row
    DROPDOWN_RECT = QRect(60, 128, 280, 175)   # the expanded list area
    DROPDOWN_ITEM_HEIGHT = 34

    OPTIONS = ["100% (Recommended)", "125%", "150%", "175%", "200%"]

    def __init__(self, initial_value: str = "200%", parent=None):
        super().__init__(parent)
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._initial_value = initial_value
        self._stage = "idle"
        self._cursor_pos = QPointF(0, 0)
        self._anim_start = None
        self._opened = False
        self._selected = False

        # Animation tick (smooth ~60 FPS)
        self._tick = QTimer(self)
        self._tick.timeout.connect(self._on_tick)
        self._tick.start(16)

        # Stage transitions handled by another timer per stage
        self._stage_timer = QTimer(self)
        self._stage_timer.setSingleShot(True)
        self._stage_timer.timeout.connect(self._advance_stage)

        # Cursor interpolation
        self._cursor_from = QPointF(20, 200)
        self._cursor_to = QPointF(20, 200)
        self._stage_elapsed = 0

        # Kick off
        QTimer.singleShot(300, lambda: self._enter_stage("idle"))

    # -----------------------------------------------------------------------
    # Stage machine
    # -----------------------------------------------------------------------
    def _enter_stage(self, stage: str):
        self._stage = stage
        self._stage_elapsed = 0

        if stage == "idle":
            self._opened = False
            self._selected = False
            self._cursor_from = QPointF(20, self.height() - 30)
            self._cursor_to = QPointF(20, self.height() - 30)

        elif stage == "move_to_combo":
            self._opened = False
            self._cursor_from = self._cursor_pos
            self._cursor_to = QPointF(
                self.COMBO_RECT.center().x() + 80,
                self.COMBO_RECT.center().y() + 4,
            )

        elif stage == "open_combo":
            self._cursor_from = self._cursor_pos
            self._cursor_to = self._cursor_pos
            # halfway through, set opened = True (in _on_tick)

        elif stage == "move_to_100":
            self._opened = True
            self._cursor_from = self._cursor_pos
            # 100% is the first item
            item_y = self.DROPDOWN_RECT.y() + self.DROPDOWN_ITEM_HEIGHT // 2 + 2
            self._cursor_to = QPointF(
                self.DROPDOWN_RECT.center().x(),
                item_y,
            )

        elif stage == "select_100":
            self._opened = True
            self._cursor_from = self._cursor_pos
            self._cursor_to = self._cursor_pos

        elif stage == "show_result":
            self._opened = False
            self._selected = True
            self._cursor_from = self._cursor_pos
            self._cursor_to = self._cursor_pos

        self._stage_timer.start(self.DURATIONS[stage])

    def _advance_stage(self):
        order = ["idle", "move_to_combo", "open_combo",
                 "move_to_100", "select_100", "show_result"]
        i = order.index(self._stage)
        next_stage = order[(i + 1) % len(order)]
        self._enter_stage(next_stage)

    # -----------------------------------------------------------------------
    # Per-frame tick — interpolates cursor & repaints
    # -----------------------------------------------------------------------
    def _on_tick(self):
        self._stage_elapsed += 16
        total = self.DURATIONS.get(self._stage, 1000)
        t = min(1.0, self._stage_elapsed / total)

        # Ease-in-out cubic
        if t < 0.5:
            ease = 4 * t * t * t
        else:
            f = 2 * t - 2
            ease = 0.5 * f * f * f + 1

        # Cursor lerp
        self._cursor_pos = QPointF(
            self._cursor_from.x() + (self._cursor_to.x() - self._cursor_from.x()) * ease,
            self._cursor_from.y() + (self._cursor_to.y() - self._cursor_from.y()) * ease,
        )

        # "Click" feedback halfway through open_combo stage
        if self._stage == "open_combo" and not self._opened and t >= 0.5:
            self._opened = True

        if self._stage == "select_100" and not self._selected and t >= 0.5:
            self._selected = True

        self.update()

    # -----------------------------------------------------------------------
    # Painting
    # -----------------------------------------------------------------------
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background of the demo area (light gray, mimics Windows Settings)
        p.fillRect(self.rect(), QColor("#f3f3f3"))

        # The white "Scale and layout" card
        card_path = QPainterPath()
        card_path.addRoundedRect(float(self.PANEL_RECT.x()),
                                 float(self.PANEL_RECT.y()),
                                 float(self.PANEL_RECT.width()),
                                 float(self.PANEL_RECT.height()),
                                 6.0, 6.0)
        p.fillPath(card_path, QColor("white"))
        p.setPen(QPen(QColor("#dadce0"), 1))
        p.drawPath(card_path)

        # "Scale" label inside the card
        p.setPen(QColor("#3c4043"))
        f = QFont("Segoe UI", 9)
        p.setFont(f)
        p.drawText(self.PANEL_RECT.x() + 14, self.PANEL_RECT.y() + 20, "Scale")

        f2 = QFont("Segoe UI", 7)
        p.setFont(f2)
        p.setPen(QColor("#5f6368"))
        p.drawText(self.PANEL_RECT.x() + 14, self.PANEL_RECT.y() + 35,
                   "Change the size of text, apps, and other items")

        # The combobox itself
        self._draw_combo(p)

        # The expanded dropdown list (when opened)
        if self._opened:
            self._draw_dropdown_list(p)

        # The cursor on top
        self._draw_cursor(p)

        # Title bar at the top of the demo, gives context
        p.setPen(QColor("#7d8a99"))
        f3 = QFont("Segoe UI", 8, QFont.Weight.Bold)
        p.setFont(f3)
        p.drawText(self.PANEL_RECT.x(), 30,
                   "Settings → System → Display")

        p.end()

    def _draw_combo(self, p: QPainter):
        r = self.COMBO_RECT

        # Selected option to show
        if self._selected:
            label = "100% (Recommended)"
            border = QColor("#34a853")  # green
            text_color = QColor("#1e7e3e")
            bg = QColor("#e8f5ec")
        elif self._opened:
            label = self._initial_value
            border = QColor("#1a73e8")  # blue (focus)
            text_color = QColor("#202124")
            bg = QColor("white")
        else:
            label = self._initial_value
            border = QColor("#dadce0")
            text_color = QColor("#202124")
            bg = QColor("white")

        # Body
        path = QPainterPath()
        path.addRoundedRect(float(r.x()), float(r.y()),
                            float(r.width()), float(r.height()),
                            4.0, 4.0)
        p.fillPath(path, bg)
        p.setPen(QPen(border, 1.5))
        p.drawPath(path)

        # Text
        p.setPen(text_color)
        f = QFont("Segoe UI", 10)
        if self._selected:
            f.setBold(True)
        p.setFont(f)
        p.drawText(r.x() + 12, r.y() + 23, label)

        # Chevron ▾ on the right
        cx = r.right() - 22
        cy = r.y() + r.height() // 2
        p.setPen(QPen(QColor("#5f6368"), 1.5))
        p.drawLine(cx - 5, cy - 2, cx, cy + 4)
        p.drawLine(cx, cy + 4, cx + 5, cy - 2)

        # If selected → checkmark next to text
        if self._selected:
            check_x = r.right() - 60
            check_y = r.y() + r.height() // 2
            p.setPen(QPen(QColor("#34a853"), 2.5))
            p.drawLine(check_x - 6, check_y, check_x - 2, check_y + 5)
            p.drawLine(check_x - 2, check_y + 5, check_x + 6, check_y - 6)

    def _draw_dropdown_list(self, p: QPainter):
        r = self.DROPDOWN_RECT

        # Drop shadow (a translucent dark rect offset down/right)
        shadow = QPainterPath()
        shadow.addRoundedRect(float(r.x() + 1), float(r.y() + 3),
                              float(r.width()), float(r.height()),
                              4.0, 4.0)
        p.fillPath(shadow, QColor(0, 0, 0, 35))

        # Body
        path = QPainterPath()
        path.addRoundedRect(float(r.x()), float(r.y()),
                            float(r.width()), float(r.height()),
                            4.0, 4.0)
        p.fillPath(path, QColor("white"))
        p.setPen(QPen(QColor("#dadce0"), 1))
        p.drawPath(path)

        # Items
        for i, opt in enumerate(self.OPTIONS):
            iy = r.y() + i * self.DROPDOWN_ITEM_HEIGHT
            item_rect = QRect(r.x(), iy, r.width(), self.DROPDOWN_ITEM_HEIGHT)

            # Is the cursor hovering this item?
            hovered = item_rect.contains(self._cursor_pos.toPoint())

            if hovered and i == 0:
                # 100% being hovered/selected
                p.fillRect(item_rect.adjusted(2, 1, -2, -1), QColor("#e8f0fe"))
                p.setPen(QColor("#1a73e8"))
                f = QFont("Segoe UI", 10, QFont.Weight.Bold)
            elif hovered:
                p.fillRect(item_rect.adjusted(2, 1, -2, -1), QColor("#f1f3f4"))
                p.setPen(QColor("#202124"))
                f = QFont("Segoe UI", 10)
            else:
                p.setPen(QColor("#202124"))
                f = QFont("Segoe UI", 10)

            p.setFont(f)
            p.drawText(item_rect.x() + 12, item_rect.y() + 22, opt)

    def _draw_cursor(self, p: QPainter):
        """Classic Windows arrow cursor."""
        x = self._cursor_pos.x()
        y = self._cursor_pos.y()

        pts = QPolygon([
            QPoint(int(x),       int(y)),
            QPoint(int(x),       int(y + 18)),
            QPoint(int(x + 5),   int(y + 14)),
            QPoint(int(x + 8),   int(y + 20)),
            QPoint(int(x + 11),  int(y + 19)),
            QPoint(int(x + 8),   int(y + 13)),
            QPoint(int(x + 13),  int(y + 13)),
        ])

        # White outline
        p.setPen(QPen(QColor("white"), 2.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPolygon(pts)

        # Black fill
        p.setPen(QPen(QColor("#202124"), 1))
        p.setBrush(QBrush(QColor("#202124")))
        p.drawPolygon(pts)

        # Click "ripple" during select_100 stage
        if self._stage == "select_100" and self._stage_elapsed < 400:
            radius = self._stage_elapsed / 8.0
            alpha = max(0, 180 - int(self._stage_elapsed / 2))
            p.setPen(QPen(QColor(26, 115, 232, alpha), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPoint(int(x), int(y)),
                          int(radius), int(radius))


# ============================================================================
# Dialog
# ============================================================================
class DpiWarningDialog(QDialog):
    SUPPRESS_KEY = "suppress_dpi_warning_mag"  # legacy, kept for back-compat

    # Context-specific intro messages
    CONTEXT_MESSAGES = {
        "mag": (
            "For <b>optimal magnification detector performance</b>, we recommend "
            "setting Windows display scaling to <b>100%</b>."
        ),
        "cellvit": (
            "For <b>accurate cell area and density measurements</b>, we highly "
            "recommend setting Windows display scaling to <b>100%</b>."
        ),
        "cellpose": (
            "For <b>accurate cell area and density measurements</b>, we "
            "recommend setting Windows display scaling to <b>100%</b>."
        ),
    }

    def __init__(self, current_scale: float, parent=None, context: str = "mag"):
        super().__init__(parent)
        self.setWindowTitle("Display Scaling Recommendation")
        self.setMinimumWidth(480)
        self.setModal(True)
        
        self._context = context

        current_pct = int(round(current_scale * 100))
        # Pick a familiar starting value for the animation
        if current_pct >= 175:
            initial = "200%"
        elif current_pct >= 137:
            initial = "150%"
        else:
            initial = "125%"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 22, 26, 18)
        layout.setSpacing(14)

        # ---------- Header ----------
        header = QLabel("🖥️  Display Scaling Recommendation")
        header.setStyleSheet("font-size: 15pt; font-weight: 600; color: #2c3e50;")
        layout.addWidget(header)

        # ---------- Intro (context-specific) ----------
        intro_msg = self.CONTEXT_MESSAGES.get(context, self.CONTEXT_MESSAGES["mag"])
        intro = QLabel(
            f"<p style='line-height:1.5;'>"
            f"{intro_msg} Your display is currently at "
            f"<b style='color:#e67e22;'>{current_pct}%</b>.<br>"
            f"<span style='color:#7d8a99; font-size:9pt;'>"
            f"Here's how to change it 👇</span></p>"
        )
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(intro)
        # ---------- Animation ----------
        self.demo = AnimatedDpiDemo(initial_value=initial)
        layout.addWidget(self.demo)

        # ---------- Open Settings button ----------
        self.btn_open = QPushButton("  Open Display Settings  ↗")
        self.btn_open.setStyleSheet(
            "QPushButton { background:#27ae60; color:white; padding:11px 16px;"
            "border:none; border-radius:6px; font-weight:600; font-size:11pt; }"
            "QPushButton:hover { background:#2ecc71; }"
            "QPushButton:pressed { background:#1e8449; }"
        )
        self.btn_open.clicked.connect(self._open_display_settings)
        layout.addWidget(self.btn_open)

        hint = QLabel(
            "<span style='color:#7d8a99; font-size:9pt;'>"
            "After changing it, restart OnSight for the change to take effect."
            "</span>"
        )
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(hint)

        # ---------- Footer ----------
        footer = QHBoxLayout()
        self.chk_suppress = QCheckBox("Don't show this again")
        self.chk_suppress.setStyleSheet("color:#555;")
        footer.addWidget(self.chk_suppress)
        footer.addStretch()

        btn_done = QPushButton("Got it")
        btn_done.setDefault(True)
        btn_done.setStyleSheet(
            "QPushButton { padding:8px 22px; border-radius:5px;"
            "background:#ecf0f1; color:#2c3e50; border:1px solid #bdc3c7; }"
            "QPushButton:hover { background:#dfe4e6; }"
        )
        btn_done.clicked.connect(self.accept)
        footer.addWidget(btn_done)
        layout.addLayout(footer)

    def is_suppress_checked(self) -> bool:
        """Return True if the user ticked the 'Don't show this again' checkbox."""
        return self.chk_suppress.isChecked()
    
    def _open_display_settings(self):
        """Open Windows Display Settings, then prompt about restarting OnSight."""
        try:
            if sys.platform == "win32":
                os.startfile("ms-settings:display")
            else:
                return
        except Exception as e:
            logging.warning(f"Could not open Display Settings: {e}")
            return
        
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(800, self._prompt_restart)


    def _prompt_restart(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("After Changing")
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setText(
            "Once you've changed the scaling to <b>100%</b> in Windows Display Settings, "
            "OnSight needs to <b>restart</b> for the new DPI to take effect."
            "<br><br>"
            "Click <b>Restart Now</b> when you've finished changing the setting."
        )
        btn_restart = msg.addButton("Restart Now", QMessageBox.ButtonRole.AcceptRole)
        btn_later = msg.addButton("Later", QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(btn_later)
        msg.exec()
        if msg.clickedButton() is btn_restart:
            self._restart_onsight()


    def _restart_onsight(self):
        import subprocess
        from PyQt6.QtWidgets import QApplication
        
        try:
            if sys.platform == "darwin" and ".app/" in sys.executable:
                app_path = sys.executable.split(".app/")[0] + ".app"
                subprocess.Popen(["open", "-n", app_path])
            else:
                subprocess.Popen([sys.executable] + sys.argv)
            
            self.accept()
            QApplication.quit()
        except Exception as e:
            logging.error(f"Failed to restart OnSight: {e}")
            QMessageBox.warning(
                self,
                "Restart Failed",
                "Could not restart OnSight automatically. "
                "Please close and reopen the app manually."
            )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def maybe_show_dpi_warning(parent=None, context: str = "mag") -> None:
    """
    Show the DPI warning dialog if:
      - We are on Windows
      - The user hasn't ticked "Don't show this again" for this context
      - The current display scaling is NOT already 100%
    
    Args:
        parent: parent widget for the dialog
        context: identifier for this warning's suppress state. Different
                 contexts have independent "don't show again" flags, so a
                 user who silenced the mag-detector warning will still see
                 the cellvit warning the first time.
    """
    if sys.platform != "win32":
        return

    settings = _load_settings()
    
    # Build a context-specific suppress key
    suppress_key = f"suppress_dpi_warning_{context}"
    
    if settings.get(suppress_key, False):
        return

    scale = _current_dpi_scale()
    if abs(scale - 1.0) < 0.05:
        return  # already 100%

    dlg = DpiWarningDialog(current_scale=scale, parent=parent, context=context)
    dlg.exec()

    if dlg.is_suppress_checked():
        settings[suppress_key] = True
        _save_settings(settings)