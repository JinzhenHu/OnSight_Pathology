from importlib import metadata

import mss
import torch
import numpy as np
import cv2
from torchvision.transforms import v2
import torch.nn as nn
import timm
import torch.nn.functional as F
from PIL import Image
from ctypes import wintypes
import ctypes
import re
from device_compat import get_device

def fix_region(region, tile_size):
    reg = region.copy()
    reg['width']  = max(reg['width'],  tile_size)
    reg['height'] = max(reg['height'], tile_size)
    return reg

def get_foreground_window_scale():
    """
    Returns the per-monitor DPI scale factor for the focused window
    (e.g. 1.0 = 100%, 1.5 = 150%). On macOS / Linux there is no
    equivalent of Windows per-monitor DPI awareness, so this always returns 1.0.
    """
    import sys
    if sys.platform != "win32":
        return 1.0

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    shcore = ctypes.WinDLL("Shcore", use_last_error=True)

    PROCESS_PER_MONITOR_DPI_AWARE = 2
    shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)

    MDT_EFFECTIVE_DPI = 0
    MONITOR_DEFAULTTONEAREST = 2

    def get_scale_for_foreground_window():
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            raise ctypes.WinError(ctypes.get_last_error())

        hMonitor = user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
        if not hMonitor:
            raise ctypes.WinError(ctypes.get_last_error())

        dpiX = wintypes.UINT()
        dpiY = wintypes.UINT()

        result = shcore.GetDpiForMonitor(
            hMonitor,
            MDT_EFFECTIVE_DPI,
            ctypes.byref(dpiX),
            ctypes.byref(dpiY)
        )
        if result != 0:
            raise OSError(f"GetDpiForMonitor failed, HRESULT={result}")

        scale_percent = dpiX.value / 96 * 100
        return {
            "hwnd": hwnd,
            "dpi": dpiX.value,
            "scale_percent": scale_percent
        }

    try:
        info = get_scale_for_foreground_window()
        return info['scale_percent'] / 100
    except Exception:
        # If anything goes wrong on Windows (no foreground window, etc.),
        # fall back to 1.0 instead of crashing the whole pipeline.
        return 1.0


def generate_tiles_nopad(img_h: int,
                         img_w: int,
                         tile: int = 1024,
                         stride: int | None = None):
    if stride is None:
        stride = tile     
    assert stride <= tile, "stride ≤ tile_size"

    y_starts = list(range(0, max(img_h - tile, 0), stride))
    if not y_starts or y_starts[-1] != img_h - tile:
        y_starts.append(img_h - tile)   
    x_starts = list(range(0, max(img_w - tile, 0), stride))
    if not x_starts or x_starts[-1] != img_w - tile:
        x_starts.append(img_w - tile)   

    return [(slice(y, y + tile), slice(x, x + tile))
            for y in y_starts
            for x in x_starts]

def process_region(region, **kwargs):
    preprocessing = v2.Compose([
        v2.ToImage(),
        #v2.Resize(size=224),
        v2.CenterCrop(size=224),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)),
    ])

    metadata = kwargs['metadata']
    model = kwargs['model']
    tile_size = metadata['tile_size']
    from custom_widgets.DpiWarningDialog import _current_dpi_scale
    os_scale = _current_dpi_scale()

    with mss.mss() as sct:
        region = fix_region(region, tile_size)
        screenshot = sct.grab(region)



    frame_orig = np.array(screenshot, dtype=np.uint8)
    #print(frame.shape)
    device = get_device()
    frame = cv2.cvtColor(frame_orig, cv2.COLOR_BGRA2RGB)
    if abs(os_scale - 1.0) > 1e-3:
        frame = cv2.resize(
            frame,
            (max(1, int(round(frame.shape[1] / os_scale))),
            max(1, int(round(frame.shape[0] / os_scale)))),
            interpolation=cv2.INTER_AREA
        )

    h, w = frame.shape[:2]

    tile_size = metadata['tile_size']         
    stride     = tile_size             
    tile_slices = generate_tiles_nopad(h, w, tile_size, stride)

    result = []
    model.eval()


    for rs, cs in tile_slices:
        tile_img = Image.fromarray(frame[rs, cs])
        tile_tensor = preprocessing(tile_img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tile_tensor)
            result.append(F.softmax(out, dim=1))

    result_tensor = torch.vstack(result)
    final_prob = torch.mean(result_tensor, dim=0)
    final_prob_numpy = final_prob.cpu().detach().numpy()

    hardcoded_mags = np.array([5.0, 10.0, 20.0, 40.0])
    
    continuous_mag = np.sum(final_prob_numpy * hardcoded_mags)

    top_3_idx = torch.argsort(final_prob, descending=True).cpu().detach().numpy()
    top_1 = top_3_idx[0]
    final_conf = float(final_prob_numpy[top_1])

    res = f"Predicted Magnification: {metadata['classes'][top_1]}<br>"
    res += f"Continuous Magnification: {continuous_mag:.2f}X"

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    probs_dict = {cls: float(p) for cls, p in zip(metadata['classes'], final_prob_numpy)}
    #print(f"Predicted Magnification: {metadata['classes'][top_1]}")
    
    metrics = {
        "conf":          float(continuous_mag),
        "pred_cls":      metadata['classes'][top_1],  
        "continuous_mag": float(continuous_mag),
        "probs":         probs_dict,
        "area_px":       frame.shape[0] * frame.shape[1],
        "mpp":           metadata.get("mpp", 0.25),
        "orig_img":      cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB),
    }
    
    return frame, res, metrics
   