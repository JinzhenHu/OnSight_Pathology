"""
device_compat.py — Cross-platform torch device selection.

Single source across Windows (CUDA),
macOS (MPS / Apple Silicon), and CPU-only systems.

Drop-in replacements for the two patterns that appear throughout the codebase:

    # cross-platform
    from device_compat import get_device, empty_cache
    device = get_device()
    empty_cache()

No behavior change on Windows or Linux GPU boxes — they still get CUDA.
On Apple Silicon they now get MPS.  On a CPU-only box they get CPU.
"""

import os
import logging

import torch


# Enable automatic CPU fallback for unimplemented MPS ops (PyTorch 1.13+).
# Some timm / cellpose ops still lack MPS kernels; without this they raise.
# Safe to set on all platforms — it's only read by the MPS backend.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _mps_available() -> bool:
    """True iff we're on macOS with a working MPS backend."""
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    try:
        return bool(backend.is_available())
    except Exception:
        return False


def get_device() -> torch.device:
    """Pick the best available torch device.

    Priority: CUDA > MPS > CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_gpu_available() -> bool:
    """True if any GPU acceleration (CUDA or MPS) is available."""
    return torch.cuda.is_available() or _mps_available()


def empty_cache() -> None:
    """Clear the active GPU's allocator cache, if any.

    No-op on CPU-only systems and safe to call from anywhere.
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif _mps_available():
            # torch.mps.empty_cache() exists in PyTorch 2.0+
            mps_mod = getattr(torch, "mps", None)
            if mps_mod is not None and hasattr(mps_mod, "empty_cache"):
                mps_mod.empty_cache()
    except Exception as e:
        logging.debug(f"empty_cache failed (non-fatal): {e}")


def device_summary() -> str:
    """Human-readable line for logs / About dialog. Example:
        'CUDA available: NVIDIA GeForce GTX 1650 (4.0 GB)'
        'MPS available (Apple Silicon)'
        'No GPU — running on CPU'
    """
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            return f"CUDA available: {props.name} ({props.total_memory / 1e9:.1f} GB)"
        except Exception:
            return "CUDA available"
    if _mps_available():
        return "MPS available (Apple Silicon)"
    return "No GPU — running on CPU"


def autocast_device_type() -> str:
    """Return the device_type string for torch.autocast.

    MPS does not support fp16 autocast as of 2.x, so we return 'cpu' on Mac
    (which effectively disables mixed precision — callers should also check
    is_amp_supported() before enabling).
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def is_amp_supported() -> bool:
    """True only on CUDA — MPS autocast is not production-ready yet."""
    return torch.cuda.is_available()
