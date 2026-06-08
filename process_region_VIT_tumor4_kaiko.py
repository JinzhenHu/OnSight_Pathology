import mss
import torch
import numpy as np
import cv2
from torchvision import transforms
import torch.nn as nn
import timm
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from torchvision.transforms import v2
from device_compat import get_device

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result
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
    #raise RuntimeError("TEST: simulated crash")
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
    additional_configs = kwargs.get('additional_configs', {})
    show_cam = additional_configs.get("Show ViT Attention Map", False)
    if isinstance(show_cam, str):
        show_cam = show_cam.lower() == 'true'

    with mss.mss() as sct:
        region = fix_region(region, tile_size)
        screenshot = sct.grab(region)


    frame_orig = np.array(screenshot, dtype=np.uint8)
    #print(frame.shape)
    device = get_device()
    frame = cv2.cvtColor(frame_orig, cv2.COLOR_BGRA2RGB)
    h, w = frame.shape[:2]

    tile_size = metadata['tile_size']         # 1024
    stride     = tile_size             
    tile_slices = generate_tiles_nopad(h, w, tile_size, stride)

    result = []
    model.eval()

    if show_cam:
        target_layers = [model.blocks[-1].norm1]
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
        full_cam = np.zeros((h, w), dtype=np.float32)
        
    for rs, cs in tile_slices:
            tile_img = Image.fromarray(frame[rs, cs])
            tile_tensor = preprocessing(tile_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                out = model(tile_tensor)
                prob = F.softmax(out, dim=1)
                result.append(prob)
                tile_conf = prob.max().item() 
            
            if show_cam:
                grayscale_cam = cam(input_tensor=tile_tensor, targets=None)
                cam_map = grayscale_cam[0]

                tile_h = rs.stop - rs.start
                tile_w = cs.stop - cs.start
                cam_map_resized = cv2.resize(cam_map, (tile_w, tile_h))
                
                weighted_cam = cam_map_resized * tile_conf
                
                full_cam[rs, cs] = np.maximum(full_cam[rs, cs], weighted_cam)

    if show_cam:
        blur_kernel = max(15, (tile_size // 8) | 1) 
        full_cam = cv2.GaussianBlur(full_cam, (blur_kernel, blur_kernel), 0)

        if full_cam.max() > 0:
            full_cam = (full_cam - full_cam.min()) / (full_cam.max() - full_cam.min() + 1e-8)




    result_tensor = torch.vstack(result)
    final_prob= torch.mean(result_tensor, dim=0)
    final_prob_numpy = torch.mean(result_tensor, dim=0).cpu().detach().numpy()
    top_3_idx = torch.argsort(final_prob, descending=True).cpu().detach().numpy()
    top_3_idx = top_3_idx[:1]
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    probs_dict = {cls: float(p) for cls, p
                in zip(metadata['classes'], final_prob_numpy)}
                
    res = ''
    for idx in top_3_idx:
        res += '{}: {:.4f}\n'.format(metadata['classes'][idx], final_prob_numpy[idx])
    
    final_conf = float(final_prob_numpy[top_3_idx[0]])
    top_1 = top_3_idx[0] 
    
    metrics = {
        "conf":      final_conf,
        "pred_cls":  metadata['classes'][top_1],  
        "probs":      probs_dict,
        "area_px":   frame.shape[0] * frame.shape[1],
        "mpp":       metadata.get("mpp", 0.25),
        "orig_img":  cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)
    }

    if show_cam:
        metrics["attention_map"] = full_cam
        
    return frame, res, metrics
   