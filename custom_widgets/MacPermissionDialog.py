"""
MacPermissionDialog.py — Premium permission dialog with per-row actions,
an animated toggle demonstration, and a restart button so users can
relaunch OnSight after granting access without finding the .app manually.

Behavior:
  - One unified dialog lists every missing permission as its own row.
  - Each row has its own "Open →" button — clicking it opens the matching
    System Settings pane and the button transforms into "Opened ✓".
  - A looping toggle animation at the bottom shows users what to do once
    they reach the Settings page.
  - Two footer buttons:
      • Restart OnSight  (primary, blue) — relaunches the app cleanly
      • Later            (secondary)     — closes; user can grant later
  - On Windows/Linux this entire module is a no-op.

Usage in app.py:
    from custom_widgets.MacPermissionDialog import check_and_prompt_permissions
    # Call AFTER WelcomeDialog has closed:
    check_and_prompt_permissions(self)
"""

import os
import sys
import subprocess

from PyQt6.QtCore import (
    Qt, QPropertyAnimation, QEasingCurve, QTimer, pyqtProperty,
)
from PyQt6.QtGui import QFont, QColor, QPainter, QBrush
from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QWidget, QGraphicsDropShadowEffect,
)

import mac_permissions


# ============================================================================
# Apple HIG color palette
# ============================================================================
COLOR_PRIMARY        = "#0A84FF"   # macOS system blue
COLOR_PRIMARY_HOVER  = "#0071E3"
COLOR_PRIMARY_PRESS  = "#005FBF"
COLOR_TEXT           = "#1D1D1F"
COLOR_TEXT_SECONDARY = "#6E6E73"
COLOR_TEXT_TERTIARY  = "#86868B"
COLOR_BG_CARD        = "#FFFFFF"
COLOR_BG_ROW         = "#F5F5F7"
COLOR_ICON_TINT      = "#E8F1FF"
COLOR_SUCCESS        = "#34C759"   # Apple green — used for the ON toggle state
COLOR_TOGGLE_OFF     = "#D1D1D6"   # Apple gray — used for the OFF toggle state
COLOR_BTN_OPEN_BG    = "#E8F1FF"   # Soft blue for inline "Open" buttons
COLOR_BTN_OPEN_HOVER = "#D5E5FF"
COLOR_BTN_DONE_BG    = "#E8E8ED"   # Muted gray for "Opened" state
COLOR_BTN_LATER_HOVER = "#EFEFF2"  # Subtle gray-hover for the Later button


# ============================================================================
# Animated macOS-style toggle widget
# ----------------------------------------------------------------------------
# Pure visual demo (not interactive). Loops OFF -> ON -> OFF forever to show
# the user exactly what they need to do once they reach the Settings page.
# ============================================================================
class _AnimatedToggleDemo(QWidget):
    """Looping animation of a macOS toggle sliding from OFF to ON and back."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Sized to approximate the real macOS toggle (which is ~52x32 in HIG)
        self.setFixedSize(56, 32)

        self._position = 0.0  # 0.0 = fully OFF (left), 1.0 = fully ON (right)

        # Animation engine
        self._anim = QPropertyAnimation(self, b"position", self)
        self._anim.setDuration(750)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._anim.finished.connect(self._on_anim_finished)

        # Kick off the loop after a small delay so the dialog has time
        # to settle visually before motion starts.
        QTimer.singleShot(500, self._start_off_to_on)

    # ---- Animation control ----
    def _start_off_to_on(self):
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.start()

    def _start_on_to_off(self):
        self._anim.setStartValue(1.0)
        self._anim.setEndValue(0.0)
        self._anim.start()

    def _on_anim_finished(self):
        # Hold at each end position briefly so users can register the state
        if self._position >= 0.99:
            QTimer.singleShot(1400, self._start_on_to_off)
        else:
            QTimer.singleShot(900, self._start_off_to_on)

    # ---- Animated property (lets QPropertyAnimation drive the value) ----
    @pyqtProperty(float)
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = value
        self.update()   # trigger paintEvent

    # ---- Painting ----
    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Interpolate background color between OFF gray and ON green
        off = QColor(COLOR_TOGGLE_OFF)
        on  = QColor(COLOR_SUCCESS)
        r = off.red()   + (on.red()   - off.red())   * self._position
        g = off.green() + (on.green() - off.green()) * self._position
        b = off.blue()  + (on.blue()  - off.blue())  * self._position
        bg = QColor(int(r), int(g), int(b))

        rect = self.rect()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(bg))
        painter.drawRoundedRect(rect, rect.height() / 2, rect.height() / 2)

        # Sliding white circle (the toggle "knob")
        margin = 3
        knob_d = rect.height() - 2 * margin
        knob_x = margin + (rect.width() - knob_d - 2 * margin) * self._position
        painter.setBrush(QBrush(QColor("white")))
        painter.drawEllipse(int(knob_x), margin, knob_d, knob_d)


# ============================================================================
# Permission row with inline action button
# ----------------------------------------------------------------------------
# Layout:  [tinted icon]  Permission Name              [ Open → ]
#                         One-line rationale text
# ============================================================================
class _PermissionRow(QWidget):
    """A single permission entry with its own 'Open' action button."""

    def __init__(self, icon_char, name, rationale, on_click, parent=None):
        super().__init__(parent)
        self._on_click = on_click
        self._has_been_opened = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 14, 12)
        layout.setSpacing(14)

        # Tinted circular icon
        icon = QLabel(icon_char)
        icon.setFixedSize(36, 36)
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet(
            f"background-color: {COLOR_ICON_TINT};"
            f"border-radius: 18px;"
            f"font-size: 18px;"
        )
        layout.addWidget(icon, 0, Qt.AlignmentFlag.AlignVCenter)

        # Text column (title + rationale)
        text_col = QVBoxLayout()
        text_col.setSpacing(1)
        text_col.setContentsMargins(0, 0, 0, 0)

        name_lbl = QLabel(name)
        name_lbl.setStyleSheet(
            f"font-size: 14px; font-weight: 600; color: {COLOR_TEXT};"
        )
        text_col.addWidget(name_lbl)

        rationale_lbl = QLabel(rationale)
        rationale_lbl.setStyleSheet(
            f"font-size: 12px; color: {COLOR_TEXT_SECONDARY};"
        )
        rationale_lbl.setWordWrap(True)
        text_col.addWidget(rationale_lbl)

        layout.addLayout(text_col, 1)

        # Inline action button — opens the relevant Settings pane
        self.btn = QPushButton("Open  →")
        self.btn.setFixedHeight(30)
        self.btn.setMinimumWidth(86)
        self.btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn.setStyleSheet(self._btn_style_default())
        self.btn.clicked.connect(self._handle_click)
        layout.addWidget(self.btn, 0, Qt.AlignmentFlag.AlignVCenter)

    # ---- Button styling variants ----
    def _btn_style_default(self):
        return f"""
            QPushButton {{
                background-color: {COLOR_BTN_OPEN_BG};
                color: {COLOR_PRIMARY};
                border-radius: 6px;
                font-size: 12px;
                font-weight: 600;
                border: none;
                padding: 0 12px;
            }}
            QPushButton:hover {{ background-color: {COLOR_BTN_OPEN_HOVER}; }}
        """

    def _btn_style_opened(self):
        return f"""
            QPushButton {{
                background-color: {COLOR_BTN_DONE_BG};
                color: {COLOR_TEXT_SECONDARY};
                border-radius: 6px;
                font-size: 12px;
                font-weight: 600;
                border: none;
                padding: 0 12px;
            }}
            QPushButton:hover {{ background-color: #DCDCE0; }}
        """

    def _handle_click(self):
        # Open the corresponding System Settings pane
        self._on_click()

        # Update button state to give visual feedback
        self._has_been_opened = True
        self.btn.setText("Opened  ✓")
        self.btn.setStyleSheet(self._btn_style_opened())
        # Button stays clickable so users can re-open if they navigated away


# ============================================================================
# Main dialog
# ============================================================================
class PermissionDialog(QDialog):
    """Frameless, premium-styled dialog listing all missing permissions."""

    def __init__(self, missing_accessibility, missing_screen_recording,
                 parent=None):
        super().__init__(parent)
        self.missing_accessibility   = missing_accessibility
        self.missing_screen_recording = missing_screen_recording

        # Frameless + translucent so we can draw rounded corners + shadow
        self.setWindowFlags(
            Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setModal(True)
        self.setFixedWidth(480)

        # Prefer SF Pro on macOS for a native feel
        if sys.platform == "darwin":
            self.setFont(QFont("SF Pro Display", 13))

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        # Outer layout reserves space around the card so the shadow can render
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 28)

        # The white card holding all content
        card = QFrame()
        card.setObjectName("card")
        card.setStyleSheet(
            f"#card {{ background-color: {COLOR_BG_CARD};"
            f"         border-radius: 14px; }}"
        )
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(40)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(0, 0, 0, 60))
        card.setGraphicsEffect(shadow)
        outer.addWidget(card)

        content = QVBoxLayout(card)
        content.setContentsMargins(32, 30, 32, 24)
        content.setSpacing(0)

        # ---- Hero icon ----
        hero = QLabel("🔐")
        hero.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hero.setStyleSheet("font-size: 42px;")
        content.addWidget(hero)
        content.addSpacing(12)

        # ---- Title ----
        title = QLabel("Permissions Required")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            f"font-size: 19px; font-weight: 600; color: {COLOR_TEXT};"
        )
        content.addWidget(title)
        content.addSpacing(6)

        # ---- Subtitle (singular/plural aware) ----
        n_missing = sum([self.missing_accessibility,
                         self.missing_screen_recording])
        word = "permission" if n_missing == 1 else "permissions"
        subtitle = QLabel(
            f"Grant the {word} below to use OnSight on macOS."
        )
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(
            f"font-size: 13px; color: {COLOR_TEXT_SECONDARY};"
        )
        content.addWidget(subtitle)
        content.addSpacing(20)

        # ---- Permission rows (one per missing permission) ----
        rows_container = QFrame()
        rows_container.setStyleSheet(
            f"background-color: {COLOR_BG_ROW}; border-radius: 10px;"
        )
        rows_layout = QVBoxLayout(rows_container)
        rows_layout.setContentsMargins(0, 4, 0, 4)
        rows_layout.setSpacing(0)

        if self.missing_accessibility:
            rows_layout.addWidget(_PermissionRow(
                "👆",
                "Accessibility",
                "Detects when you click to select a region.",
                mac_permissions.open_accessibility_settings,
            ))
        if self.missing_screen_recording:
            rows_layout.addWidget(_PermissionRow(
                "🖥️",
                "Screen Recording",
                "Reads pixels from the region you select.",
                mac_permissions.open_screen_recording_settings,
            ))

        content.addWidget(rows_container)
        content.addSpacing(22)

        # ---- Animated toggle demo (visual instruction) ----
        demo_caption = QLabel(
            "In System Settings, toggle ON.\n"
            "Then come back and click Restart below."
        )
        demo_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        demo_caption.setStyleSheet(
            f"font-size: 12px; color: {COLOR_TEXT_SECONDARY}; line-height: 1.4;"
        )
        content.addWidget(demo_caption)
        content.addSpacing(10)

# Mock-up of how the toggle row looks in macOS System Settings.
        # We render the app name on the left and the animated toggle on the
        # right, mimicking the real Settings UI so users know which row to
        # interact with.
        toggle_row = QHBoxLayout()
        toggle_row.addStretch()

        # Container mimicking a single Settings row
        settings_row = QFrame()
        settings_row.setStyleSheet(
            f"background-color: {COLOR_BG_ROW};"
            f"border-radius: 8px;"
        )
        settings_row_layout = QHBoxLayout(settings_row)
        settings_row_layout.setContentsMargins(14, 8, 14, 8)
        settings_row_layout.setSpacing(14)

        # App icon + name on the left
        app_icon = QLabel("🔬")
        app_icon.setStyleSheet("font-size: 18px;")
        settings_row_layout.addWidget(app_icon)

        app_name = QLabel("OnSightPathology_App")
        app_name.setStyleSheet(
            f"font-size: 13px; font-weight: 500; color: {COLOR_TEXT};"
        )
        settings_row_layout.addWidget(app_name)

        settings_row_layout.addStretch()

        # The animated toggle on the right
        settings_row_layout.addWidget(_AnimatedToggleDemo())

        toggle_row.addWidget(settings_row)
        toggle_row.addStretch()
        content.addLayout(toggle_row)
        content.addSpacing(22)

        # ---- Footer buttons: Later (secondary) + Restart (primary) ----
        button_row = QHBoxLayout()
        button_row.setSpacing(10)

        later_btn = QPushButton("Later")
        later_btn.setFixedHeight(40)
        later_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        later_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLOR_TEXT_SECONDARY};
                border-radius: 8px;
                font-size: 14px;
                font-weight: 500;
                border: none;
                padding: 0 16px;
            }}
            QPushButton:hover {{
                background-color: {COLOR_BTN_LATER_HOVER};
                color: {COLOR_TEXT};
            }}
        """)
        later_btn.clicked.connect(self.reject)
        button_row.addWidget(later_btn, 1)

        restart_btn = QPushButton("Restart OnSight")
        restart_btn.setFixedHeight(40)
        restart_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        restart_btn.setDefault(True)
        restart_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_PRIMARY};
                color: white;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 600;
                border: none;
                padding: 0 16px;
            }}
            QPushButton:hover  {{ background-color: {COLOR_PRIMARY_HOVER}; }}
            QPushButton:pressed {{ background-color: {COLOR_PRIMARY_PRESS}; }}
        """)
        restart_btn.clicked.connect(self._restart_onsight)
        button_row.addWidget(restart_btn, 2)

        content.addLayout(button_row)

    # ------------------------------------------------------------------
    # Restart logic — mirrors app.py::_set_ui_scale relaunch behaviour.
    # ------------------------------------------------------------------
    def _restart_onsight(self):
        """Relaunch OnSight cleanly so newly-granted permissions take effect.

        macOS permission changes are only applied when the process restarts —
        toggling Screen Recording in Settings while OnSight is running has no
        effect until relaunch.

        For a frozen .app bundle, we use `open -n <App.app>` so macOS spawns a
        new instance (preserves Dock icon, menu bar, and permission grants).
        For dev/source runs, we re-exec the current Python interpreter with
        the same argv.
        """
        try:
            if sys.platform == "darwin" and ".app/" in sys.executable:
                # Inside a PyInstaller .app bundle — find the .app path and
                # ask macOS to launch a fresh instance.
                app_path = sys.executable.split(".app/")[0] + ".app"
                subprocess.Popen(["open", "-n", app_path])
            else:
                # Source / dev run — re-exec the same interpreter.
                subprocess.Popen([sys.executable] + sys.argv)
        except Exception:
            # If restart fails for any reason, fall through and just quit.
            # The user can relaunch manually from the Dock.
            pass

        # Close dialog and exit the current process — the new instance will
        # already be starting up.
        self.accept()
        QApplication.quit()


# ============================================================================
# Public entry point
# ============================================================================
def check_and_prompt_permissions(parent=None):
    """Check macOS permissions and show a single dialog if any are missing.

    No-op on Windows/Linux. Designed to be called after WelcomeDialog has
    closed so the two dialogs don't overlap on first launch.

    Returns True if all permissions were already granted (no dialog shown),
    False if the dialog was displayed.
    """
    if not mac_permissions.is_macos():
        return True

    missing_a11y   = not mac_permissions.check_accessibility_permission()
    missing_screen = not mac_permissions.check_screen_recording_permission()

    if not (missing_a11y or missing_screen):
        return True

    dlg = PermissionDialog(missing_a11y, missing_screen, parent)
    dlg.exec()
    return False