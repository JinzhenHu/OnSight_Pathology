import torch
import yaml
from retinanet.Model import MyMitosisDetection, resource_path
import mss
import numpy as np
import cv2
from retinanet.utlis.detection_helper import create_anchors, process_output, rescale_boxes
from retinanet.utlis.nms_WSI import nms, nms_patch
from ultralytics.utils.plotting import Annotator
import itertools
import math
def _safe_float(value, default=0.2):
    """
    Return float(value) if it is a valid numeric string/number,
    otherwise return the default.
    """
    try:
        # Accept None, empty string, non-numeric strings, etc.
        return float(value)
    except (TypeError, ValueError):
        return default
    
def fix_region(region, tile_size):
    reg = region.copy()
    reg['width']  = max(reg['width'],  tile_size)
    reg['height'] = max(reg['height'], tile_size)
    return reg

def process_region(region, **kwargs):

    device = torch.device("cuda")

    thred = 0.1

    Cl = 1-_safe_float(kwargs['additional_configs'].get('sensitivity', 1))

    metadata = kwargs['metadata']
    model = kwargs['model']
    

    tile_size = metadata['tile_size']

    with mss.mss() as sct:
        region = fix_region(region, tile_size)
        screenshot = sct.grab(region)

    frame_orig = np.array(screenshot, dtype=np.uint8)
    frame = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)


    stride = 384  #
    all_detections = []
    all_magnification =[]

    h, w = frame.shape[:2]
    num_x = math.ceil((w - tile_size) / stride) + 1
    num_y = math.ceil((h - tile_size) / stride) + 1

    #if the crop is smaller, upscale it to tile_size
    if h < tile_size or w<tile_size:
        frame = cv2.resize(frame,(tile_size,tile_size))

    for i in range(num_x):
        for j in range(num_y):
            x_start = i * stride
            y_start = j * stride

            # if the crop past the edge, shift it back so it fits
            if x_start + tile_size > w:
                x_start = w - tile_size
            if y_start + tile_size > h:
                y_start = h - tile_size

            tile = frame[y_start:y_start + tile_size,
                        x_start:x_start + tile_size,
                        :]

            box = model.process_image(tile)
            if box.size == 0:
                continue

            # Adjust box coordinates to the global frame
            box_coords = np.atleast_2d(box)
            global_boxes = box_coords.copy()
            global_boxes[:, 0] += x_start
            global_boxes[:, 1] += y_start
            global_boxes[:, 2] += x_start
            global_boxes[:, 3] += y_start
            all_detections.append(global_boxes.tolist())

    # Apply Non-Maximum Suppression on all detections
    prediction = nms(list(itertools.chain(*all_detections)))
    if frame.shape[0]> tile_size:
        BIG_FONT   = 34   # pixels
        THICK_LINE = 4    # pixels
    if frame.shape[0]<= tile_size:
        BIG_FONT   = 24   # pixels
        THICK_LINE = 2    # pixels
    # Set up the annotator for drawing boxes
    annotator = Annotator(
        frame,
        line_width=THICK_LINE,
        font_size=BIG_FONT,
        font="Arial_Bold.ttf",
        pil=False,
        example="abc",
    )

    # Annotate detections that meet the confidence threshold
    for row in prediction:
        if row[5] >= Cl:
            # Choose color based on confidence
            if row[5] > 0.7:
                base_color = (0, 255, 0)
            elif row[5] > 0.4:
                base_color = (255, 0, 0)
            else:
                base_color = (0, 0, 255)
            label_text = f'score {row[5]*100:.1f} %'
            annotator.box_label(
                row[:4],
                label=label_text,
                color=base_color,
                txt_color=(255, 255, 255),
                rotated=False,
            )

    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]


    annotated_result = annotator.result()
    annotated_frame = annotated_result.astype(np.uint8)

    annotated_frame=   cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    prediction = [row for row in prediction if row[5] >= Cl]
    txt = f"Number of mitosis detected in this frame: {len(prediction)}"
    metrics = {
        "mitosis": len(prediction),
        "area_px": frame.shape[0] * frame.shape[1],
        "mpp": metadata.get("mpp", 0.25),
        "orig_img": cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)
    }
    return annotated_frame, txt, metrics
