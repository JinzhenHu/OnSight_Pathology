import mss
import torch
import numpy as np
import cv2
from torchvision import transforms
import torch.nn as nn
import timm
import torch.nn.functional as F
from PIL import Image

def process_region(region, **kwargs):
    preprocessing = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    metadata = kwargs['metadata']
    model = kwargs['model']
    tile_size = metadata['tile_size']

    with mss.mss() as sct:
        screenshot = sct.grab(region)


    frame = np.array(screenshot, dtype=np.uint8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame[:max((frame.shape[0] // tile_size) * tile_size, tile_size),
              :max((frame.shape[1] // tile_size) * tile_size, tile_size), :]
    h, w = frame.shape[:2]
    n_rows, n_cols = h // tile_size, w // tile_size

    result = []
    model.eval()

    if n_rows >= 1 and n_cols >= 1:
        for i in range(n_rows):
            for j in range(n_cols):
                tile = frame[
                    i*tile_size:(i+1)*tile_size,
                    j*tile_size:(j+1)*tile_size
                ]
                tile = Image.fromarray(tile)
                tile = preprocessing(tile).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(tile)
                    result.append(F.softmax(out, dim=1))
    else:
        # region small case resie back to tilesize
        big_tile = cv2.resize(frame, (tile_size, tile_size))
        big_tile = Image.fromarray(big_tile)
        big_tile = preprocessing(big_tile).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(big_tile)
            result.append(F.softmax(out, dim=1))

    result_tensor = torch.vstack(result)
    final_prob= torch.mean(result_tensor, dim=0)
    final_prob_numpy = torch.mean(result_tensor, dim=0).cpu().detach().numpy()
    top_3_idx = torch.argsort(final_prob, descending=True).cpu().detach().numpy()
    top_3_idx = top_3_idx[:3]
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


    res = ''
    for idx in top_3_idx:
        res += '{}: {:.4f}\n'.format(metadata['classes'][idx], final_prob_numpy[idx])


    return frame, res

