import mss
import numpy as np
import cv2

from utils import extract_tiles
def fix_region(region, tile_size):
    reg = region.copy()
    reg['width']  = max(reg['width'],  tile_size)
    reg['height'] = max(reg['height'], tile_size)
    return reg

# Log-based correction (more forgiving for small # of tiles)
def correction_factor(n_tiles):
    return max(1.0 - 0.15 * np.log1p(n_tiles), 0.5)

def process_region(region, **kwargs):

    metadata = kwargs['metadata']
    model = kwargs['model']

    ###

    tile_size = metadata['tile_size']

    with mss.mss() as sct:
        region = fix_region(region, tile_size)
        screenshot = sct.grab(region)

    frame = np.array(screenshot, dtype=np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    frame = frame[:max((frame.shape[0]//tile_size)*tile_size, tile_size), :max((frame.shape[1]//tile_size)*tile_size, tile_size), :]

    slices = extract_tiles(frame, tile_size)


    import matplotlib.colors as mcolors
    from detectron2.data import MetadataCatalog
    from detectron2.utils.visualizer import ColorMode, Visualizer
    from detectron2.utils.memory import retry_if_cuda_oom

    # classes = ["Positive", "Negative", "Misc"]  # NOTE: for now just pos, neg, misc
    positive_idx = 0
    negative_idx = 1
    colormaps = [mcolors.to_rgba('r'), mcolors.to_rgba('b'), mcolors.to_rgba('g')]
    MetadataCatalog.get("dataset").set(thing_classes=metadata['classes'], thing_colors=colormaps)
    metadata = MetadataCatalog.get("dataset")



    tile_size_y = tile_size_x = tile_size
    if frame.shape[0] < tile_size:
        tile_size_y = frame.shape[0]
    if frame.shape[1] < tile_size:
        tile_size_x = frame.shape[1]

    num_pos = 0
    num_pos_and_neg = 0

    seg_mask = np.zeros(frame.shape)
    k = 0
    for i in range(frame.shape[0] // tile_size_y):
        for j in range(frame.shape[1] // tile_size_x):


            # Note: Batch processing doesn't work...have to process individually
            outputs = retry_if_cuda_oom(model)(slices[k])

            v = Visualizer(slices[k][:, :, ::-1], metadata, scale=1, instance_mode=ColorMode.SEGMENTATION)
            instances = outputs['instances'].to("cpu")

            if len(instances.get('pred_classes')):

                for kk in range(len(instances)):
                    out = v.draw_binary_mask(np.array(instances.get('pred_masks')[kk].detach()),
                                             color=colormaps[instances.get('pred_classes')[kk]])
                curr_mask = out.get_image()  # it is RGB uint8 0~255
                curr_mask = curr_mask[:, :, ::-1]

                seg_mask[
                (i * tile_size):((i * tile_size) + tile_size),
                (j * tile_size):((j * tile_size) + tile_size),
                :
                ] = curr_mask


                _num_pos = sum(instances.get('pred_classes') == positive_idx).item()
                _num_pos_and_neg = _num_pos + sum(instances.get('pred_classes') == negative_idx).item()


            else:
                # no masks found
                _num_pos = _num_pos_and_neg = 0

                seg_mask[
                (i * tile_size):((i * tile_size) + tile_size),
                (j * tile_size):((j * tile_size) + tile_size),
                :
                ] = slices[k]

            num_pos += _num_pos
            num_pos_and_neg += _num_pos_and_neg

            k += 1


    try:
        _correction_factor = float(kwargs['additional_configs'].get('correction_factor', 0.15))
    except:
        _correction_factor = 0.15

    num_neg = num_pos_and_neg - num_pos
    num_neg = int(num_neg * correction_factor(len(slices)))
    positivity = num_pos / (num_pos + num_neg) * 100 if (num_pos + num_neg) > 0 else 0

    text = '(+) {:.2f} %\n'.format(positivity)
    text += '(+) cells: {}\n'.format(num_pos)
    text += '(-) cells: {}\n'.format(num_neg)

    return seg_mask.astype(np.uint8), text