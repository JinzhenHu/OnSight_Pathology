"""
mac_permissions.py — macOS Accessibility & Screen Recording permission helpers.

On macOS, unsigned apps need explicit user authorization to:
  1. Listen to global mouse events (pynput) — requires "Accessibility"
  2. Capture screen pixels (mss)             — requires "Screen Recording"
"""

import sys
import subprocess
import logging


def is_macos() -> bool:
    return sys.platform == "darwin"


def check_accessibility_permission() -> bool:
    return False
    """Return True if this process has macOS Accessibility permission.

    Uses AXIsProcessTrusted from the ApplicationServices framework. This
    is a silent check — it does NOT prompt the user. Always returns True
    on non-macOS platforms so callers don't need to branch.
    """
    if not is_macos():
        return True
    try:
        from ApplicationServices import AXIsProcessTrusted
        return bool(AXIsProcessTrusted())
    except ImportError:
        logging.warning(
            "[mac_permissions] pyobjc not installed; cannot check "
            "Accessibility permission. Assuming granted."
        )
        return True
    except Exception as e:
        logging.warning(f"[mac_permissions] Accessibility check failed: {e}")
        return True  # fail-open so we don't block the user incorrectly


def check_screen_recording_permission() -> bool:
    return False
    """Return True if this process has Screen Recording permission.

    Uses CGPreflightScreenCaptureAccess (macOS 10.15+ Catalina), which
    queries permission state WITHOUT triggering the system prompt. This
    is critical: we want OnSight's own dialog to appear first, not
    macOS's generic "OnSight wants to record screen" alert.
    """
    if not is_macos():
        return True

    try:
        from Quartz import CGPreflightScreenCaptureAccess
        return bool(CGPreflightScreenCaptureAccess())
    except ImportError:
        logging.warning(
            "[mac_permissions] CGPreflightScreenCaptureAccess unavailable "
            "(macOS < 10.15 or Quartz not installed); assuming granted."
        )
        return True
    except Exception as e:
        logging.warning(
            f"[mac_permissions] CGPreflightScreenCaptureAccess failed: {e}; "
            f"assuming granted."
        )
        return True  # fail-open, never block user incorrectly


def open_accessibility_settings() -> None:
    """Open System Settings to the Accessibility privacy pane."""
    if not is_macos():
        return
    url = "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
    subprocess.Popen(["open", url])


def open_screen_recording_settings() -> None:
    """Open System Settings to the Screen Recording privacy pane.

    DESIGN NOTE: We deliberately do NOT call CGRequestScreenCaptureAccess
    here. That API surfaces a macOS system dialog ("OnSight wants to
    record screen"), which appears on top of our own dialog and confuses
    users — they see TWO permission prompts at once.

    OnSight should already be present in the Screen Recording list as a
    side effect of normal app initialization (Qt/mss touch screen-related
    APIs during startup, which registers the app with TCC even without an
    explicit prompt). If a user reports being unable to find OnSight in
    the Settings list, calling CGRequestScreenCaptureAccess once would
    force it to appear — but the tradeoff isn't worth the double-prompt UX.
    """
    if not is_macos():
        return
    url = "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"
    subprocess.Popen(["open", url])


# ============================================================================
# User-facing dialog text. Kept here so wording is easy to tweak in one place.
# ============================================================================

ACCESSIBILITY_DIALOG_TITLE = "Accessibility Permission Required"
ACCESSIBILITY_DIALOG_BODY = (
    "OnSight uses mouse event capture to let you select regions on your "
    "screen. macOS requires explicit permission for this.\n\n"
    "Steps to grant permission:\n"
    "  1. Click 'Open Settings' below\n"
    "  2. Find 'OnSight' (or 'Terminal' / 'Python' in dev mode) in the list\n"
    "  3. Toggle it ON\n"
    "  4. Click 'Restart OnSight' below\n\n"
    "Without this permission, region selection will hang indefinitely."
)

SCREEN_RECORDING_DIALOG_TITLE = "Screen Recording Permission Required"
SCREEN_RECORDING_DIALOG_BODY = (
    "OnSight needs to capture pixels from your screen to run analysis. "
    "macOS requires explicit permission for this.\n\n"
    "Steps to grant permission:\n"
    "  1. Click 'Open Settings' below\n"
    "  2. Find 'OnSight' (or 'Terminal' / 'Python' in dev mode) in the list\n"
    "  3. Toggle it ON\n"
    "  4. Click 'Restart OnSight' below\n\n"
    "Without this permission, captured regions will appear black."
)