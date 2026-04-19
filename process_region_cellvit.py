from importlib import metadata
import numpy as np
import mss
import cv2
import math
import torch
from skimage.segmentation import clear_border
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms as T

def _safe_float(value, default=0.2):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def fix_region(region, tile_size):
    reg = region.copy()
    reg['width'] = max(reg['width'], tile_size)
    reg['height'] = max(reg['height'], tile_size)
    return reg

def pad_to_multiple(img_np, multiple=256):
    h, w = img_np.shape[:2]
    new_h = math.ceil(h / multiple) * multiple
    new_w = math.ceil(w / multiple) * multiple

    pad_bottom = new_h - h
    pad_right = new_w - w

    img_pad = cv2.copyMakeBorder(
        img_np,
        0,
        pad_bottom,
        0,
        pad_right,
        borderType=cv2.BORDER_REFLECT_101,
    )
    return img_pad, h, w

def preprocess_image(img, mean, std, device):

    img_pad, orig_h, orig_w = pad_to_multiple(img, multiple=256)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

    x = transform(Image.fromarray(img_pad)).unsqueeze(0).to(device)
    return img, img_pad, x, orig_h, orig_w

def colorize_instance_map(instance_map):
    h, w = instance_map.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    ids = np.unique(instance_map)
    ids = ids[ids > 0]

    rng = np.random.default_rng(42)
    colors = {i: rng.integers(50, 255, size=3, dtype=np.uint8) for i in ids}

    for i in ids:
        canvas[instance_map == i] = colors[i]

    return canvas

def draw_contours_on_image(image_rgb, instance_dict, type_id_to_name):
    vis = image_rgb.copy()
    color_map = {
        1: (255, 0, 0),
        2: (34, 221, 77),
        3: (35, 92, 236),
        4: (254, 255, 0),
        5: (255, 159, 68),
    }

    for inst_id, info in instance_dict.items():
        contour = np.array(info["contour"], dtype=np.int32)
        t = int(info["type"])
        color = color_map.get(t, (255, 255, 255))

        if contour.ndim == 2 and contour.shape[0] >= 3:
            cv2.drawContours(vis, [contour], -1, color, 2)

        cx, cy = info["centroid"]
        label = type_id_to_name.get(t, str(t))
        cv2.putText(
            vis,
            label,
            (int(cx), int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA
        )
    return vis
def draw_contours_on_image(image_rgb, instance_dict, type_id_to_name):
    vis = image_rgb.copy()
    color_map = {
        1: (255, 0, 0),
        2: (34, 221, 77),
        3: (35, 92, 236),
        4: (254, 255, 0),
        5: (255, 159, 68),
    }

    for inst_id, info in instance_dict.items():
        contour = np.array(info["contour"], dtype=np.int32)
        t = int(info["type"])
        color = color_map.get(t, (255, 255, 255))

        if contour.ndim == 2 and contour.shape[0] >= 3:
            cv2.drawContours(vis, [contour], -1, color, 2)

        cx, cy = info["centroid"]
        
        base_label = type_id_to_name.get(t, str(t))
        
        prob = info.get("type_prob")
        if prob is not None:
            label = f"{base_label} {prob*100:.0f}%" 
        else:
            label = base_label

        text_x = int(cx) - 10
        text_y = int(cy) - 5

        cv2.putText(
            vis,
            label,
            (max(0, text_x), max(0, text_y)), 
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA
        )
    return vis
def process_region(region, **kwargs):
    metadata = kwargs['metadata']
    model_list = kwargs['model']
    
    configs = kwargs.get('additional_configs', {})
    
    tile_size = metadata.get('tile_size', 256) 

    with mss.mss() as sct:
        region = fix_region(region, tile_size)
        screenshot = sct.grab(region)

    frame_orig = np.array(screenshot, dtype=np.uint8)
    frame = cv2.cvtColor(frame_orig, cv2.COLOR_BGRA2RGB)

    h, w = frame.shape[:2]
    model, mean, std, type_id_to_name, tissue_id_to_name, device = model_list
    img_np, img_pad, x, orig_h, orig_w = preprocess_image(frame, mean, std, device)

    
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=(device.type == "cuda")
    ):
        out = model(x, retrieve_tokens=True)

    out["nuclei_binary_map"] = out["nuclei_binary_map"][:, :, :orig_h, :orig_w]
    out["hv_map"] = out["hv_map"][:, :, :orig_h, :orig_w]
    out["nuclei_type_map"] = out["nuclei_type_map"][:, :, :orig_h, :orig_w]

    tissue_logits = out["tissue_types"]
    tissue_prob = F.softmax(tissue_logits, dim=-1)[0].detach().cpu().numpy()
    tissue_id = int(np.argmax(tissue_prob))
    tissue_name = tissue_id_to_name[tissue_id] if tissue_id_to_name else str(tissue_id)

    out["nuclei_binary_map"] = F.softmax(out["nuclei_binary_map"], dim=1)
    out["nuclei_type_map"] = F.softmax(out["nuclei_type_map"], dim=1)

    instance_map_tensor, instance_type_list = model.calculate_instance_map(
        out, magnification=40
    )
    instance_map = instance_map_tensor[0].cpu().numpy().astype(np.int32)
    instance_map = clear_border(instance_map)
    instance_dict = instance_type_list[0]

    remaining_ids = set(np.unique(instance_map))

    filtered_instance_dict = {}
    for inst_id, info in instance_dict.items():

        if int(inst_id) in remaining_ids and int(inst_id) != 0:
            filtered_instance_dict[inst_id] = info

    instance_color = colorize_instance_map(instance_map)
    overlay = draw_contours_on_image(img_np, filtered_instance_dict, type_id_to_name)

    final_vis_img_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    metrics = {
        "orig_img": cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB),
        "area_px": frame.shape[0] * frame.shape[1],
        "mpp": metadata.get("mpp", 0.25),
    }
    text = ""
    
    return final_vis_img_bgr.astype(np.uint8), text, metrics