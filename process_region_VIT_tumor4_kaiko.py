import mss
import torch
import numpy as np
import cv2
from torchvision import transforms
import torch.nn as nn
import timm
import torch.nn.functional as F
from PIL import Image
def fix_region(region, tile_size):
    reg = region.copy()
    reg['width']  = max(reg['width'],  tile_size)
    reg['height'] = max(reg['height'], tile_size)
    return reg

def generate_tiles_nopad(img_h: int,
                         img_w: int,
                         tile: int = 1024,
                         stride: int | None = None):
    if stride is None:
        stride = tile     # 没重叠
    assert stride <= tile, "stride ≤ tile_size"

    # 1. 行起点
    y_starts = list(range(0, max(img_h - tile, 0), stride))
    if not y_starts or y_starts[-1] != img_h - tile:
        y_starts.append(img_h - tile)   # 把最后一块顶到下边缘
    # 2. 列起点
    x_starts = list(range(0, max(img_w - tile, 0), stride))
    if not x_starts or x_starts[-1] != img_w - tile:
        x_starts.append(img_w - tile)   # 顶到右边缘

    # 3. 组合 slice
    return [(slice(y, y + tile), slice(x, x + tile))
            for y in y_starts
            for x in x_starts]

def process_region(region, **kwargs):
    preprocessing = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    metadata = kwargs['metadata']
    model = kwargs['model']
    tile_size = metadata['tile_size']

    with mss.mss() as sct:
        region = fix_region(region, tile_size)
        screenshot = sct.grab(region)


    frame = np.array(screenshot, dtype=np.uint8)
    #print(frame.shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    h, w = frame.shape[:2]

    tile_size = metadata['tile_size']         # 1024
    stride     = tile_size               # 可调：无重叠 = tile_size
    tile_slices = generate_tiles_nopad(h, w, tile_size, stride)

    result = []
    model.eval()
    for rs, cs in tile_slices:
        tile_img = Image.fromarray(frame[rs, cs])
        tile_tensor = preprocessing(tile_img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tile_tensor)
            result.append(F.softmax(out, dim=1))





    # frame = frame[:max((frame.shape[0] // tile_size) * tile_size, tile_size),
    #           :max((frame.shape[1] // tile_size) * tile_size, tile_size), :]
    # h, w = frame.shape[:2]
    # n_rows, n_cols = h // tile_size, w // tile_size

    # result = []
    # model.eval()

    # if n_rows >= 1 and n_cols >= 1:
    #     for i in range(n_rows):
    #         for j in range(n_cols):
    #             tile = frame[
    #                 i*tile_size:(i+1)*tile_size,
    #                 j*tile_size:(j+1)*tile_size
    #             ]
    #             tile = Image.fromarray(tile)
    #             tile = preprocessing(tile).unsqueeze(0).to(device)
    #             with torch.no_grad():
    #                 out = model(tile)
    #                 result.append(F.softmax(out, dim=1))


    result_tensor = torch.vstack(result)
    final_prob= torch.mean(result_tensor, dim=0)
    final_prob_numpy = torch.mean(result_tensor, dim=0).cpu().detach().numpy()
    top_3_idx = torch.argsort(final_prob, descending=True).cpu().detach().numpy()
    top_3_idx = top_3_idx[:1]
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


    res = ''
    for idx in top_3_idx:
        res += '{}: {:.4f}\n'.format(metadata['classes'][idx], final_prob_numpy[idx])


    return frame, res

