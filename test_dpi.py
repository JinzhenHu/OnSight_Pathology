"""
Test whether mss.grab() returns the same pixel count regardless of
the calling process's DPI awareness mode.

Usage:
    python test_mss_dpi.py
    
It spawns two child processes — one DPI-Aware and one DPI-Unaware —
and has each of them grab the same 1024x1024 screen region, then
prints the shape of what each got back.
"""
import sys
import subprocess
import ctypes

# ─────────────────────────────────────────────────────────────
# Child mode: just do the grab and report shape
# ─────────────────────────────────────────────────────────────
if len(sys.argv) > 1 and sys.argv[1] == "--child":
    mode = sys.argv[2]  # "aware" or "unaware"

    if mode == "aware":
        # Set Per-Monitor V2 DPI awareness BEFORE importing anything visual
        try:
            ctypes.windll.user32.SetProcessDpiAwarenessContext(
                ctypes.c_void_p(-4)
            )
            print(f"[{mode}] DPI awareness set to Per-Monitor V2")
        except Exception as e:
            print(f"[{mode}] Failed to set DPI awareness: {e}")
    else:
        # Explicitly stay DPI-Unaware (Windows default for non-aware apps)
        # We simply don't call any SetProcessDpiAwareness API.
        print(f"[{mode}] Staying DPI-Unaware (default)")

    import mss
    import numpy as np

    region = {"left": 100, "top": 100, "width": 1024, "height": 1024}
    print(f"[{mode}] Requesting region: {region}")

    with mss.mss() as sct:
        # Report what mss thinks the monitors are
        for i, mon in enumerate(sct.monitors):
            print(f"[{mode}] Monitor[{i}]: {mon}")

        screenshot = sct.grab(region)
        arr = np.array(screenshot)
        print(f"[{mode}] Grabbed shape: {arr.shape}")
        print(f"[{mode}] Grabbed size (mss): {screenshot.size}")

    # Also report what Windows thinks the screen size is from THIS process's view
    user32 = ctypes.windll.user32
    print(f"[{mode}] GetSystemMetrics(0,1) = "
          f"({user32.GetSystemMetrics(0)}, {user32.GetSystemMetrics(1)})")
    
    sys.exit(0)


# ─────────────────────────────────────────────────────────────
# Parent mode: spawn both children and compare
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print(" mss DPI awareness test")
print("=" * 70)

# Report parent process info
user32 = ctypes.windll.user32
shcore = ctypes.windll.shcore
try:
    awareness = ctypes.c_int()
    shcore.GetProcessDpiAwareness(0, ctypes.byref(awareness))
    awareness_names = {0: "Unaware", 1: "System Aware", 2: "Per-Monitor"}
    print(f"Parent process awareness: "
          f"{awareness_names.get(awareness.value, awareness.value)}")
except Exception as e:
    print(f"Couldn't query parent awareness: {e}")

print()
print("Running DPI-AWARE child ...")
print("-" * 70)
subprocess.run([sys.executable, __file__, "--child", "aware"])

print()
print("Running DPI-UNAWARE child ...")
print("-" * 70)
# Use a fresh subprocess that doesn't inherit any DPI flags from us
subprocess.run([sys.executable, __file__, "--child", "unaware"])

print()
print("=" * 70)
print(" Compare the 'Grabbed shape' lines above.")
print(" If they're identical → mss always returns physical pixels.")
print(" If different → mss adapts to caller's DPI awareness.")
print("=" * 70)