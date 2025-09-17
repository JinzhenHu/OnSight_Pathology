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
    scale_factor = h / tile_size  
    # font_scale = max(0.5, 0.8 * scale_factor)
    # thickness = max(1, int(2 * scale_factor))
    font_scale = max(0.8, 1.3 * scale_factor)  
    thickness = max(2, int(3 * scale_factor))
#############################################################  
#Old
#############################################################        
    # Set up the annotator for drawing boxes
    # annotator = Annotator(
    #     frame,
    #     line_width=THICK_LINE,
    #     font_size=BIG_FONT,
    #     font="Arial_Bold.ttf",
    #     pil=False,
    #     example="abc",
    # )

    # # Annotate detections that meet the confidence threshold
    # for row in prediction:
    #     if row[5] >= Cl:
    #         # Choose color based on confidence
    #         if row[5] > 0.7:
    #             base_color = (0, 255, 0)
    #         elif row[5] > 0.4:
    #             base_color = (255, 0, 0)
    #         else:
    #             base_color = (0, 0, 255)
    #         label_text = f'score {row[5]*100:.1f} %'
    #         annotator.box_label(
    #             row[:4],
    #             label=label_text,
    #             color=base_color,
    #             txt_color=(255, 255, 255),
    #             rotated=False,
    #         )

    # # Get frame dimensions
    # frame_height, frame_width = frame.shape[:2]
    # annotated_result = annotator.result()
    # annotated_frame = annotated_result.astype(np.uint8)

#############################################################  
#New
#############################################################  
    annotated_frame = frame.copy()

    # --- Professional Annotation with OpenCV ---
    # You can adjust these parameters for style
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font_scale = 0.6
    # font_thickness = 1
    text_color = (255, 255, 255) # White

    for row in prediction:
        if row[5] >= Cl: # Check against confidence threshold
            # Define box color based on confidence score
            score = row[5]
            if score > 0.7:
                box_color = (0, 255, 0)  # Green for high confidence
            elif score > 0.4:
                box_color = (255, 165, 0) # Yellow for medium confidence
            else:
                box_color = (0, 0, 255)  # Red for low confidence

            # Get box coordinates
            x1, y1, x2, y2 = map(int, row[:4])

            # 1. Draw the bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, thickness)

            # 2. Prepare the label text and size
            label_text = f'{(score * 100):.0f}%'
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            
            # Adjusted position to prevent text from being cut off at the top
            label_y_start = y1 - text_h - 10
            if label_y_start < 0:
                label_y_start = y1 + 5 # Move label inside if it goes off-screen
            
            cv2.rectangle(annotated_frame, (x1, label_y_start), (x1 + text_w, label_y_start + text_h + 5), box_color, -1)

            # 4. Draw the label text using dynamic scale and thickness
            text_y_pos = label_y_start + text_h
            cv2.putText(annotated_frame, label_text, (x1, text_y_pos), font, font_scale, text_color, thickness, cv2.LINE_AA)


    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]



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
