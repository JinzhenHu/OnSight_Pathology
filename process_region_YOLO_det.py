import mss
import numpy as np
import cv2

from utils import extract_tiles
def fix_region(region, tile_size):
    reg = region.copy()
    reg['width']  = max(reg['width'],  tile_size)
    reg['height'] = max(reg['height'], tile_size)
    return reg

def process_region(region, **kwargs):

    metadata = kwargs['metadata']
    model = kwargs['model']

    ###

    tile_size = metadata['tile_size']

    with mss.mss() as sct:
        region = fix_region(region, tile_size)
        screenshot = sct.grab(region)

    frame_orig = np.array(screenshot, dtype=np.uint8)
    frame = cv2.cvtColor(frame_orig, cv2.COLOR_BGRA2BGR)
    frame = frame[:max((frame.shape[0]//tile_size)*tile_size, tile_size), :max((frame.shape[1]//tile_size)*tile_size, tile_size), :]



    slices = extract_tiles(frame, tile_size)

    # Detect and render

    try:
        _conf = 1-float(kwargs['additional_configs'].get('sensitivity', 0.4))
    except:
        _conf = 0.6

    results = model(slices, conf=_conf)

    from ultralytics.utils.plotting import Annotator
    from ultralytics.data.augment import LetterBox
    import torch

    try:
        _box_size = abs(int(kwargs['additional_configs'].get('box_size (0 for full size)', 0)))
    except:
        _box_size = 0



    tile_size_y = tile_size_x = tile_size
    if frame.shape[0] < tile_size:
        tile_size_y = frame.shape[0]
    if frame.shape[1] < tile_size:
        tile_size_x = frame.shape[1]

    seg_mask = np.zeros(frame.shape)
    k = 0
    for i in range(frame.shape[0] // tile_size_y):
        for j in range(frame.shape[1] // tile_size_x):
            # need to override so that i can choose box colors...
            if results[k].boxes is not None and results[k].boxes.shape[0] != 0:
                annotator = Annotator(
                    np.ascontiguousarray(results[k].orig_img),
                    line_width=None,
                    font_size=None,
                    font="Arial.ttf",
                    pil=False,
                    example=results[0].names,
                )

                colors = {
                    0: (92, 92, 240),  # posistive. red
                    1: (255, 191, 0),  # negative. blue
                    2: (0, 255, 0),  # misc. green
                }

                for _, d in enumerate(reversed(results[k].boxes)):
                    label = None
                    box = d.xyxy.squeeze()

                    # 0 is full sized box. custom is to make it look more like a dot
                    if _box_size:
                        box = box.cpu().numpy()
                        box = np.array([
                            (box[0] + ((box[2] - box[0]) / 2)) - _box_size,
                            (box[1] + ((box[3] - box[1]) / 2)) - _box_size,
                            (box[2] - ((box[2] - box[0]) / 2)) + _box_size,
                            (box[3] - ((box[3] - box[1]) / 2)) + _box_size,
                        ])

                    annotator.box_label(
                        box,
                        label,
                        color=colors[int(d.cls.item())],
                        rotated=False,
                    )

                seg_mask[
                (i * tile_size):((i * tile_size) + tile_size),
                (j * tile_size):((j * tile_size) + tile_size),
                :
                # ] = results[k].plot(labels=False, boxes=False)
                ] = annotator.result()

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

    num_pos = 0
    num_pos_neg = 0
    for r in results:
        num_pos_curr = torch.sum(r.boxes.cls == 0).cpu().numpy()
        num_pos += num_pos_curr
        num_pos_neg += num_pos_curr + torch.sum(r.boxes.cls == 1).cpu().numpy()

    text = '(+) {:.2f} %\n'.format(num_pos / num_pos_neg * 100 if num_pos_neg > 0 else 0)
    text += '(+) cells: {}\n'.format(num_pos)
    text += '(-) cells: {}\n'.format(num_pos_neg - num_pos)
    metrics = {
        "mib_pos":   int(num_pos),
        "mib_total": int(num_pos_neg),
        "area_px":   frame.shape[0] * frame.shape[1],
        "mpp":       metadata.get("mpp", 0.25),
        "orig_img": cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)
    }
    return seg_mask.astype(np.uint8), text, metrics
