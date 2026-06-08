"""
mac_permissions.py — macOS Accessibility & Screen Recording permission helpers.

On macOS, unsigned apps need explicit user authorization to:
  1. Listen to global mouse events (pynput) — requires "Accessibility"
  2. Capture screen pixels (mss)             — requires "Screen Recording"

Without these, OnSight's region selection hangs and captured frames are black.
This module provides:
  - Non-blocking permission checks (no system prompt shown)
  - One-call functions to open System Settings directly to the right pane
  - User-facing dialog text

On Windows/Linux all functions are no-ops returning True.
"""

import sys
import subprocess
import logging


def is_macos() -> bool:
    return sys.platform == "darwin"


def check_accessibility_permission() -> bool:
    """Return True if this process has macOS Accessibility permission.

    Uses AXIsProcessTrusted from the ApplicationServices framework. This
    is a silent check — it does NOT prompt the user. Always returns True
    on non-macOS platforms so callers don't need to branch.
    """
    if not is_macos():
        return True
    try:
        from ApplicationServices import AXIsProcessTrusted
        #return bool(AXIsProcessTrusted())
        return False
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
    """Return True if this process can actually capture screen pixels.

    macOS doesn't expose a clean API for this, so we test empirically:
    grab a tiny 20x20 region with mss and check whether it's all-black.
    A pure-zero result means the OS denied capture.
    """
    if not is_macos():
        return True
    try:
        import mss
        import numpy as np
        with mss.mss() as sct:
            shot = sct.grab({"left": 0, "top": 0, "width": 20, "height": 20})
            arr = np.array(shot)
            # All zeros → permission denied. Any non-zero pixel → granted.
            #return bool(arr.sum() > 0)
            return False
    except Exception as e:
        logging.warning(f"[mac_permissions] Screen recording check failed: {e}")
        return True  # fail-open


def open_accessibility_settings() -> None:
    """Open System Settings to the Accessibility privacy pane."""
    if not is_macos():
        return
    url = "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
    subprocess.Popen(["open", url])


def open_screen_recording_settings() -> None:
    """Open System Settings to the Screen Recording privacy pane."""
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
    "  4. Restart OnSight\n\n"
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
    "  4. Restart OnSight\n\n"
    "Without this permission, captured regions will appear black."
)
