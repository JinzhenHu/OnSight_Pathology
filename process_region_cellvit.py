# -*- coding: utf-8 -*-
"""
Realtime screen-capture CellViT-SAM-H inference.
"""
import os
import math
import json

import mss
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from PIL import Image
from skimage.segmentation import clear_border
from skimage.measure import regionprops_table
from torchvision import transforms as T


# ============================================================
# Config
# ============================================================

PATCH_SIZE = 256

# 192 = overlap 64.
STRIDE = 192

DEDUP_DIST_THRESH = 10.0

PAD_VALUE_RGB = (255, 255, 255)

USE_MIXED_PRECISION = True


COLOR_TABLE_RGB = {
    0: (0, 0, 0),
    1: (255, 0, 0),       # Neoplastic
    2: (34, 221, 77),     # Inflammatory
    3: (35, 92, 236),     # Connective
    4: (254, 255, 0),     # Dead
    5: (255, 159, 68),    # Epithelial
}

HEX_TABLE = {
    1: "#ff0000",
    2: "#22dd4d",
    3: "#235cec",
    4: "#feff00",
    5: "#ff9f44",
}


# ============================================================
# Basic helpers
# ============================================================

def _safe_float(value, default=0.2):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fix_region(region, tile_size):
    reg = region.copy()
    reg["width"] = max(reg["width"], tile_size)
    reg["height"] = max(reg["height"], tile_size)
    return reg


def build_transform(mean, std):
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


# ============================================================
# Sliding window helpers
# ============================================================

def sliding_windows(height, width, patch_size, stride):
    if height <= patch_size:
        ys = [0]
    else:
        ys = list(range(0, height - patch_size + 1, stride))
        last_y = height - patch_size
        if ys[-1] != last_y:
            ys.append(last_y)

    if width <= patch_size:
        xs = [0]
    else:
        xs = list(range(0, width - patch_size + 1, stride))
        last_x = width - patch_size
        if xs[-1] != last_x:
            xs.append(last_x)

    coords = []
    for y0 in ys:
        for x0 in xs:
            coords.append((x0, y0))

    return coords


def pad_patch_to_256_center(patch_rgb, target_size=256):
    h, w = patch_rgb.shape[:2]

    if h == target_size and w == target_size:
        return patch_rgb, 0, 0, w, h

    if h > target_size or w > target_size:
        raise ValueError(f"Patch larger than target size. Got H={h}, W={w}")

    pad_top = (target_size - h) // 2
    pad_bottom = target_size - h - pad_top
    pad_left = (target_size - w) // 2
    pad_right = target_size - w - pad_left

    padded = cv2.copyMakeBorder(
        patch_rgb,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=PAD_VALUE_RGB,
    )

    return padded, pad_left, pad_top, w, h


# ============================================================
# CellViT inference on one patch
# ============================================================

@torch.no_grad()
def run_one_cellvit_patch(
    model,
    patch_rgb_256,
    transform,
    device,
    magnification=40,
):
    x = transform(Image.fromarray(patch_rgb_256)).unsqueeze(0).to(device)

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=(USE_MIXED_PRECISION and device.type == "cuda"),
    ):
        out = model(x, retrieve_tokens=True)

    preds = {
        "nuclei_binary_map": F.softmax(out["nuclei_binary_map"], dim=1),
        "hv_map": out["hv_map"],
        "nuclei_type_map": F.softmax(out["nuclei_type_map"], dim=1),
    }

    instance_map_tensor, instance_type_list = model.calculate_instance_map(
        preds,
        magnification=magnification,
    )

    instance_map = instance_map_tensor[0].detach().cpu().numpy().astype(np.int32)
    instance_types = instance_type_list[0]

    return instance_map, instance_types


# ============================================================
# Patch cells -> global frame cells
# ============================================================

def contour_area(contour):
    contour = np.asarray(contour, dtype=np.int32)
    if contour.ndim != 2 or len(contour) < 3:
        return 0.0
    return float(abs(cv2.contourArea(contour)))


def center_in_safe_region(cx, cy, patch_w, patch_h, margin=16):
    if patch_w <= 2 * margin or patch_h <= 2 * margin:
        return True

    return (
        cx >= margin
        and cy >= margin
        and cx < patch_w - margin
        and cy < patch_h - margin
    )


def patch_cells_to_global_cells(
    instance_types,
    patch_x0,
    patch_y0,
    pad_offset_x,
    pad_offset_y,
    valid_w,
    valid_h,
    whole_w,
    whole_h,
    type_id_to_name,
):
    cells = []

    for instance_id, info in instance_types.items():
        if "contour" not in info or "centroid" not in info:
            continue

        contour_pad = np.asarray(info["contour"], dtype=np.int32)
        centroid_pad = np.asarray(info["centroid"], dtype=np.float32)

        if contour_pad.ndim != 2 or contour_pad.shape[1] != 2:
            continue

        if len(contour_pad) < 3:
            continue

        cx_pad = float(centroid_pad[0])
        cy_pad = float(centroid_pad[1])

        # padded-patch coords -> actual local patch coords
        cx_local = cx_pad - pad_offset_x
        cy_local = cy_pad - pad_offset_y

        # filter objects whose centroid is in padded area
        if cx_local < 0 or cy_local < 0:
            continue
        if cx_local >= valid_w or cy_local >= valid_h:
            continue

        # local patch coords -> full frame coords
        cx_global = cx_local + patch_x0
        cy_global = cy_local + patch_y0

        if cx_global < 0 or cy_global < 0:
            continue
        if cx_global >= whole_w or cy_global >= whole_h:
            continue

        contour_local = contour_pad.copy()
        contour_local[:, 0] -= pad_offset_x
        contour_local[:, 1] -= pad_offset_y

        contour_local[:, 0] = np.clip(contour_local[:, 0], 0, valid_w - 1)
        contour_local[:, 1] = np.clip(contour_local[:, 1], 0, valid_h - 1)

        contour_global = contour_local.copy()
        contour_global[:, 0] += patch_x0
        contour_global[:, 1] += patch_y0

        contour_global[:, 0] = np.clip(contour_global[:, 0], 0, whole_w - 1)
        contour_global[:, 1] = np.clip(contour_global[:, 1], 0, whole_h - 1)

        area = contour_area(contour_global)
        if area < 3:
            continue

        label_id = int(info.get("type", -1))
        label_prob = float(info.get("type_prob", 0.0))

        cell = {
            "patch_instance_id": int(instance_id),
            "label_id": label_id,
            "label_name": type_id_to_name.get(label_id, f"class_{label_id}"),
            "label_prob": label_prob,
            "centroid_x": float(cx_global),
            "centroid_y": float(cy_global),
            "contour": contour_global,
            "area": area,
            "patch_x0": int(patch_x0),
            "patch_y0": int(patch_y0),
            "keep_center_safe": center_in_safe_region(
                cx_local,
                cy_local,
                valid_w,
                valid_h,
                margin=16,
            ),
        }

        cells.append(cell)

    return cells


# ============================================================
# Deduplication
# ============================================================

def choose_better_cell(a, b):
    if a["keep_center_safe"] and not b["keep_center_safe"]:
        return a

    if b["keep_center_safe"] and not a["keep_center_safe"]:
        return b

    if a["label_prob"] > b["label_prob"]:
        return a

    if b["label_prob"] > a["label_prob"]:
        return b

    if a["area"] >= b["area"]:
        return a

    return b


def deduplicate_cells(cells, dist_thresh=10.0):
    kept = []

    for cell in cells:
        duplicate_idx = None

        for i, old in enumerate(kept):
            dx = cell["centroid_x"] - old["centroid_x"]
            dy = cell["centroid_y"] - old["centroid_y"]
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < dist_thresh:
                duplicate_idx = i
                break

        if duplicate_idx is None:
            kept.append(cell)
        else:
            kept[duplicate_idx] = choose_better_cell(kept[duplicate_idx], cell)

    return kept

def draw_contours_on_image(image_rgb, cells, draw_text=False, thickness=2):
    vis = image_rgb.copy()

    for cell in cells:
        contour = np.asarray(cell["contour"], dtype=np.int32)
        label_id = int(cell["label_id"])
        color = COLOR_TABLE_RGB.get(label_id, (255, 255, 255))

        if contour.ndim == 2 and contour.shape[0] >= 3:
            cv2.drawContours(vis, [contour], -1, color, thickness)

        if draw_text:
            cx = int(round(cell["centroid_x"]))
            cy = int(round(cell["centroid_y"]))
            text = str(label_id)
            cv2.putText(
                vis,
                text,
                (cx + 1, cy - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.25,
                color,
                max(1, thickness),
                cv2.LINE_AA,
            )

    return vis


# ============================================================
# Build global instance map
# ============================================================

def build_global_instance_map(cells, height, width):
    instance_map = np.zeros((height, width), dtype=np.int32)

    id_to_type = {}

    # Large cells first, small cells later can overwrite tiny overlaps.
    cells_sorted = sorted(cells, key=lambda c: c["area"], reverse=True)

    for global_id, cell in enumerate(cells_sorted, start=1):
        contour = np.asarray(cell["contour"], dtype=np.int32)

        if contour.ndim != 2 or contour.shape[1] != 2 or len(contour) < 3:
            continue

        cv2.fillPoly(instance_map, [contour], int(global_id))
        id_to_type[int(global_id)] = int(cell["label_id"])
        cell["global_id"] = int(global_id)

    return instance_map, id_to_type, cells_sorted


# ============================================================
# Drawing
# ============================================================

def draw_contours_on_image(image_rgb, cells, draw_text=False, thickness=2):
    vis = image_rgb.copy()

    for cell in cells:
        contour = np.asarray(cell["contour"], dtype=np.int32)
        label_id = int(cell["label_id"])
        color = COLOR_TABLE_RGB.get(label_id, (255, 255, 255))

        if contour.ndim == 2 and contour.shape[0] >= 3:
            cv2.drawContours(vis, [contour], -1, color, thickness)

        if draw_text:
            cx = int(round(cell["centroid_x"]))
            cy = int(round(cell["centroid_y"]))
            text = str(label_id)
            cv2.putText(
                vis,
                text,
                (cx + 1, cy - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.25,
                color,
                max(1, thickness),
                cv2.LINE_AA,
            )

    return vis


# ============================================================
# HTML output
# ============================================================

def build_stats_html_and_metrics(
    frame_rgb,
    instance_map,
    cells,
    type_id_to_name,
    mpp,
    h,
    w,
    metrics,
):
    total_cells = len(cells)

    if total_cells == 0 or instance_map.max() == 0:
        empty_html = """
        <div style='background-color:#ffffff; padding:20px; text-align:center; font-family:Arial, sans-serif; color:#666;'>
            <span style='font-size:12pt;'>No cells detected in this region.</span>
        </div>
        """
        metrics["cell_count"] = 0
        metrics["cellularity_percent"] = 0.0
        return empty_html, metrics

    gray_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

    props = regionprops_table(
        instance_map,
        intensity_image=gray_frame,
        properties=["label", "area", "eccentricity", "intensity_mean"],
    )

    df = pd.DataFrame(props)

    global_id_to_type = {}
    for cell in cells:
        gid = cell.get("global_id", None)
        if gid is not None:
            global_id_to_type[int(gid)] = int(cell["label_id"])

    df["type_id"] = df["label"].map(global_id_to_type)
    df = df.dropna(subset=["type_id"])
    df["type_id"] = df["type_id"].astype(int)

    # px^2 -> μm^2
    df["area"] = df["area"] * (mpp ** 2)

    class_stats = df.groupby("type_id").agg({
        "label": "count",
        "area": ["mean", "std"],
        "eccentricity": ["mean", "std"],
        "intensity_mean": ["mean", "std"],
    }).fillna(0)

    metrics["cell_count"] = int(total_cells)

    tissue_area_mm2 = (h * w * (mpp ** 2)) / 1_000_000.0
    metrics["cellularity_percent"] = total_cells / tissue_area_mm2 if tissue_area_mm2 > 0 else 0

    html = f"""
    <div style='background-color:#ffffff; padding:10px; font-family: "Segoe UI", Arial, sans-serif; color:#222;'>
        <div style='margin-bottom:12px; font-size:12pt;'>
            <b style='color:#000;'>Total Cells:</b> {total_cells}
            <span style='color:#ccc; margin:0 10px;'>|</span>
            <b style='color:#000;'>Density:</b> {metrics['cellularity_percent']:.0f} / mm²
        </div>

        <table width='100%' cellspacing='0' cellpadding='8'
               style='font-size:11pt; text-align:left; border-collapse:collapse; border-top:2px solid #333; border-bottom:2px solid #333;'>
            <tr style='border-bottom:1px solid #333;'>
                <th style='padding-left:10px;'>Class</th>
                <th>Count</th>
                <th>Area <span style='font-size:9pt; font-weight:normal; color:#666;'>(μm²)</span></th>
                <th>Eccen.</th>
                <th>Intensity <span style='font-size:9pt; font-weight:normal; color:#666;'>(Mean)</span></th>
            </tr>
    """

    for t_id, t_name in type_id_to_name.items():
        t_id = int(t_id)
        t_name = str(t_name)

        if t_id == 0 or "background" in t_name.lower():
            continue

        color_hex = HEX_TABLE.get(t_id, "#999999")

        row_str = "<tr style='border-bottom:1px solid #f0f0f0;'>"
        row_str += "<td style='padding-left:10px;'>"
        row_str += f"<span style='color:{color_hex}; font-size:14pt; vertical-align:middle; margin-right:6px;'>■</span>"
        row_str += f"<span style='vertical-align:middle;'>{t_name}</span></td>"

        if t_id in class_stats.index:
            stats = class_stats.loc[t_id]

            count = int(stats[("label", "count")])
            a_m, a_s = stats[("area", "mean")], stats[("area", "std")]
            e_m, e_s = stats[("eccentricity", "mean")], stats[("eccentricity", "std")]
            i_m, i_s = stats[("intensity_mean", "mean")], stats[("intensity_mean", "std")]

            metrics[f"{t_name}_Count"] = count
            metrics[f"{t_name}_Area_mean"] = float(a_m)
            metrics[f"{t_name}_Eccen_mean"] = float(e_m)
            metrics[f"{t_name}_Intensity_mean"] = float(i_m)

            row_str += f"<td>{count}</td>"
            row_str += f"<td>{a_m:.1f} <span style='color:#888; font-size:9pt;'>±{a_s:.1f}</span></td>"
            row_str += f"<td>{e_m:.2f} <span style='color:#888; font-size:9pt;'>±{e_s:.2f}</span></td>"
            row_str += f"<td>{i_m:.1f} <span style='color:#888; font-size:9pt;'>±{i_s:.1f}</span></td>"
        else:
            row_str += "<td style='color:#ccc;'>0</td>"
            row_str += "<td style='color:#ccc;'>-</td><td style='color:#ccc;'>-</td><td style='color:#ccc;'>-</td>"

        row_str += "</tr>"
        html += row_str

    html += "</table></div>"

    return html, metrics


# ============================================================
# Main realtime process_region
# ============================================================

def process_region(region, **kwargs):
    metadata = kwargs["metadata"]
    model_list = kwargs["model"]
    configs = kwargs.get("additional_configs", {})

    tile_size = int(metadata.get("tile_size", PATCH_SIZE))
    mpp = _safe_float(configs.get("mpp", metadata.get("mpp", 0.25)), 0.25)
    magnification = int(_safe_float(configs.get("magnification", 40), 40))
    contour_thickness = int(_safe_float(configs.get("contour_thickness", 2), 2))

    # Allow UI/json override
    stride = int(_safe_float(configs.get("stride", STRIDE), STRIDE))
    dedup_dist = _safe_float(configs.get("dedup_dist", DEDUP_DIST_THRESH), DEDUP_DIST_THRESH)
    draw_text = bool(configs.get("draw_text", False))

    with mss.mss() as sct:
        region = fix_region(region, tile_size)
        screenshot = sct.grab(region)

    frame_orig = np.array(screenshot, dtype=np.uint8)
    frame_rgb = cv2.cvtColor(frame_orig, cv2.COLOR_BGRA2RGB)

    h, w = frame_rgb.shape[:2]

    model, mean, std, type_id_to_name, tissue_id_to_name, device = model_list
    transform = build_transform(mean, std)

    metrics = {
        "orig_img": frame_rgb.copy(),
        "area_px": h * w,
        "mpp": mpp,
        "cell_count": 0,
        "cellularity_percent": 0.0,
    }

    coords = sliding_windows(
        height=h,
        width=w,
        patch_size=PATCH_SIZE,
        stride=stride,
    )

    all_cells = []

    for patch_idx, (x0, y0) in enumerate(coords, start=1):
        patch_rgb = frame_rgb[
            y0:min(y0 + PATCH_SIZE, h),
            x0:min(x0 + PATCH_SIZE, w),
        ]

        patch_256, pad_offset_x, pad_offset_y, valid_w, valid_h = pad_patch_to_256_center(
            patch_rgb,
            target_size=PATCH_SIZE,
        )


        _, instance_types = run_one_cellvit_patch(
            model=model,
            patch_rgb_256=patch_256,
            transform=transform,
            device=device,
            magnification=magnification,
        )


        patch_cells = patch_cells_to_global_cells(
            instance_types=instance_types,
            patch_x0=x0,
            patch_y0=y0,
            pad_offset_x=pad_offset_x,
            pad_offset_y=pad_offset_y,
            valid_w=valid_w,
            valid_h=valid_h,
            whole_w=w,
            whole_h=h,
            type_id_to_name=type_id_to_name,
        )

        all_cells.extend(patch_cells)

    cells = deduplicate_cells(
        all_cells,
        dist_thresh=dedup_dist,
    )

    instance_map, id_to_type, cells = build_global_instance_map(
        cells,
        height=h,
        width=w,
    )

    # clear_border after global stitching
    instance_map = clear_border(instance_map)

    remaining_ids = set(np.unique(instance_map))
    remaining_ids.discard(0)

    cells = [
        c for c in cells
        if int(c.get("global_id", -1)) in remaining_ids
    ]

    overlay_rgb = draw_contours_on_image(
        frame_rgb,
        cells,
        draw_text=draw_text,
        thickness=contour_thickness,
    )

    final_vis_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

    html, metrics = build_stats_html_and_metrics(
        frame_rgb=frame_rgb,
        instance_map=instance_map,
        cells=cells,
        type_id_to_name=type_id_to_name,
        mpp=mpp,
        h=h,
        w=w,
        metrics=metrics,
    )

    return final_vis_bgr.astype(np.uint8), html, metrics

