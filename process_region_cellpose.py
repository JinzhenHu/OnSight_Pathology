import mss
import numpy as np
import cv2
from skimage.color import rgb2lab

from utils import extract_tiles


def fix_region(region, tile_size):
    reg = region.copy()
    reg['width'] = max(reg['width'], tile_size)
    reg['height'] = max(reg['height'], tile_size)
    return reg


def visualize_ki67_overlay(img, masks, labels, ki67_positive, alpha=0.4):
    """
    Color each nucleus red (positive) or green (negative).
    """
    # Start from a copy of the original
    overlay = img.copy()

    # Make sure we work in BGR for drawing
    # (Assuming img is BGR from OpenCV)
    for lab, is_pos in zip(labels, ki67_positive):
        region = (masks == lab)

        if is_pos:
            color = (0, 0, 255)  # red in BGR
        else:
            color = (255, 0, 0)  # blue in BGR

        overlay[region] = color

    blended = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    return blended


def classify_ki67_labb_approach(img, masks, min_area=10, threshold=0):
    def compute_lab_b_signal(img_rgb, masks, labels, min_area=10):
        lab = rgb2lab(img_rgb)
        b_channel = lab[..., 2]  # b*: brown = positive, blue = negative

        scores = []
        valid_labels = []
        for lab_id in labels:
            region = masks == lab_id
            if region.sum() < min_area:
                continue
            scores.append(float(b_channel[region].mean()))
            valid_labels.append(lab_id)

        return np.array(valid_labels), np.array(scores)

    labels = np.unique(masks)
    labels = labels[labels != 0]

    valid_labels, scores = compute_lab_b_signal(img, masks, labels, min_area)

    return valid_labels, scores > threshold

def um2_to_px(area_um2, mpp):
        return area_um2 / (mpp ** 2)

def process_region(region, **kwargs):
    metadata = kwargs['metadata']
    model = kwargs['model']

    ###

    tile_size = metadata['tile_size']

    with mss.mss() as sct:
        region = fix_region(region, tile_size)
        screenshot = sct.grab(region)

    frame_orig = np.array(screenshot, dtype=np.uint8)
    frame = cv2.cvtColor(frame_orig, cv2.COLOR_BGRA2RGB)  # cellpose-sam takes in rgb!
    frame = frame[:max((frame.shape[0] // tile_size) * tile_size, tile_size),
            :max((frame.shape[1] // tile_size) * tile_size, tile_size), :]

    slices = extract_tiles(frame, tile_size)

    # Detect and render
    masks, _, _ = model.eval(slices)

    try:
        _mask_alpha = 1 - float(kwargs['additional_configs'].get('mask_transparency', 0.5))
        _positivity_thresh = float(kwargs['additional_configs'].get('positivity_thresh', 0))
        _min_area = float(kwargs['additional_configs'].get('min_area_um2', 1))
    except:
        _mask_alpha = 0.5
        _positivity_thresh = 0
        _min_area = 1

    # NOTE: Default µm² area will be using 20x
    _min_area = um2_to_px(_min_area, 0.504)

    tile_size_y = tile_size_x = tile_size
    if frame.shape[0] < tile_size:
        tile_size_y = frame.shape[0]
    if frame.shape[1] < tile_size:
        tile_size_x = frame.shape[1]

    num_pos = 0
    num_pos_neg = 0
    seg_mask = np.zeros(frame.shape)
    k = 0
    for i in range(frame.shape[0] // tile_size_y):
        for j in range(frame.shape[1] // tile_size_x):

            if masks[k].max() > 0:

                labels, ki67_pos = classify_ki67_labb_approach(
                    slices[k],
                    masks[k],
                    min_area=_min_area,
                    threshold=_positivity_thresh,
                )

                num_pos += ki67_pos.sum()
                num_pos_neg += len(ki67_pos)

                # Opencv stuff in BGR
                overlayed_img = visualize_ki67_overlay(cv2.cvtColor(slices[k], cv2.COLOR_RGB2BGR), masks[k], labels,
                                                       ki67_pos, alpha=_mask_alpha)

                seg_mask[
                (i * tile_size):((i * tile_size) + tile_size),
                (j * tile_size):((j * tile_size) + tile_size),
                :
                ] = overlayed_img

            else:
                # no masks found
                seg_mask[
                (i * tile_size):((i * tile_size) + tile_size),
                (j * tile_size):((j * tile_size) + tile_size),
                :
                ] = frame[
                    (i * tile_size):((i * tile_size) + tile_size),
                    (j * tile_size):((j * tile_size) + tile_size),
                    :
                    ]

            k += 1

    text = '(+) {:.2f} %\n'.format(num_pos / num_pos_neg * 100 if num_pos_neg > 0 else 0)
    text += '(+) cells: {}\n'.format(num_pos)
    text += '(-) cells: {}\n'.format(num_pos_neg - num_pos)
    metrics = {
        "mib_pos": int(num_pos),
        "mib_total": int(num_pos_neg),
        "area_px": frame.shape[0] * frame.shape[1],
        "mpp": metadata.get("mpp", 0.25),
        "orig_img": cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)
    }
    return seg_mask.astype(np.uint8), text, metrics
