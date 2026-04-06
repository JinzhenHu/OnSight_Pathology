import mss
import torch
import numpy as np
import cv2
from torchvision import transforms
import torch.nn as nn
import timm
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2
from pytorch_grad_cam import GradCAMPlusPlus

# 把你的 reshape_transform 放到最外层
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
                         tile: int = 512,
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
            #v2.Resize(256, interpolation=v2.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    metadata = kwargs['metadata']
    model = kwargs['model']
    additional_configs = kwargs.get('additional_configs', {})

    # 🚀 改进 1：动态读取 UI 开关，控制 CAM 是否计算
    show_cam = additional_configs.get("Show ViT Attention Map", False)
    if isinstance(show_cam, str):
        show_cam = show_cam.lower() == 'true'

    tile_size = metadata['tile_size']

    with mss.mss() as sct:
        region = fix_region(region, tile_size)
        screenshot = sct.grab(region)

    frame_orig = np.array(screenshot, dtype=np.uint8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = cv2.cvtColor(frame_orig, cv2.COLOR_BGRA2RGB)
    h, w = frame.shape[:2]

    stride = tile_size               # 可调：无重叠 = tile_size
    tile_slices = generate_tiles_nopad(h, w, tile_size, stride)

    result = []
    model.eval()

    # 🚀 改进 2：只有开启了 CAM 才会初始化对象，大幅节省显存和计算资源
    if show_cam:
        target_layers = [model.blocks[-1].norm1]
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
        full_cam = np.zeros((h, w), dtype=np.float32)

    for rs, cs in tile_slices:
        tile_img = Image.fromarray(frame[rs, cs])
        tile_tensor = preprocessing(tile_img).unsqueeze(0).to(device)
        
        # 1. 正常分类预测 (永远执行)
        with torch.no_grad():
            out = model(tile_tensor)
            prob = F.softmax(out, dim=1)
            result.append(prob)
            
            # 拿到当前 tile 的最高概率
            tile_conf = prob.max().item() 
        
        # 🚀 改进 3：只有打开了开关，才跑这段贼慢的 CAM 逻辑
        if show_cam:
            grayscale_cam = cam(input_tensor=tile_tensor, targets=None)
            cam_map = grayscale_cam[0]

            tile_h = rs.stop - rs.start
            tile_w = cs.stop - cs.start

            cam_map_resized = cv2.resize(cam_map, (tile_w, tile_h))
            
            # 直接按 tile 置信度缩放，保留高亮度区域
            weighted_cam = cam_map_resized * tile_conf
            
            # 使用 np.maximum 覆盖，防止最后一行/一列强制裁剪叠加时造成的异常黑块
            full_cam[rs, cs] = np.maximum(full_cam[rs, cs], weighted_cam)

    # 🚀 改进 4：使用大核全局高斯模糊彻底“融化”切块的硬边缘
    if show_cam:
        # 自适应模糊核大小：随切块大小成比例放大，并确保为奇数
        blur_kernel = max(15, (tile_size // 8) | 1) 
        full_cam = cv2.GaussianBlur(full_cam, (blur_kernel, blur_kernel), 0)
        
        # 归一化到 0.0 - 1.0 之间
        if full_cam.max() > 0:
            full_cam = (full_cam - full_cam.min()) / (full_cam.max() - full_cam.min() + 1e-8)


    result_tensor = torch.vstack(result)
    final_prob = torch.mean(result_tensor, dim=0)
    final_prob_numpy = final_prob.cpu().detach().numpy()
    top_3_idx = torch.argsort(final_prob, descending=True).cpu().detach().numpy()
    top_3_idx = top_3_idx[:1]
    
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    probs_dict = {cls: float(p) for cls, p in zip(metadata['classes'], final_prob_numpy)}
                
    res = ''
    for idx in top_3_idx:
        res += '{}: {:.4f}\n'.format(metadata['classes'][idx], final_prob_numpy[idx])
        
    final_conf = float(final_prob_numpy[top_3_idx[0]])
    top_1 = top_3_idx[0] 
    
    metrics = {
        "conf":      final_conf,
        "pred_cls":  metadata['classes'][top_1],  
        "probs":     probs_dict,
        "area_px":   frame.shape[0] * frame.shape[1],
        "mpp":       metadata.get("mpp", 0.25),
        "orig_img":  cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB),
    }
    
    # 只有开关打开时，才会向主 UI 吐出 attention_map 数据
    if show_cam:
        metrics["attention_map"] = full_cam
        
    return frame, res, metrics
    # for rs, cs in tile_slices:
    #     tile_img = Image.fromarray(frame[rs, cs])
    #     tile_tensor = preprocessing(tile_img).unsqueeze(0).to(device)
    #     with torch.no_grad():
    #         out = model(tile_tensor)
    #         result.append(F.softmax(out, dim=1))





    # # frame = frame[:max((frame.shape[0] // tile_size) * tile_size, tile_size),
    # #           :max((frame.shape[1] // tile_size) * tile_size, tile_size), :]
    # # h, w = frame.shape[:2]
    # # n_rows, n_cols = h // tile_size, w // tile_size

    # # result = []
    # # model.eval()

    # # if n_rows >= 1 and n_cols >= 1:
    # #     for i in range(n_rows):
    # #         for j in range(n_cols):
    # #             tile = frame[
    # #                 i*tile_size:(i+1)*tile_size,
    # #                 j*tile_size:(j+1)*tile_size
    # #             ]
    # #             tile = Image.fromarray(tile)
    # #             tile = preprocessing(tile).unsqueeze(0).to(device)
    # #             with torch.no_grad():
    # #                 out = model(tile)
    # #                 result.append(F.softmax(out, dim=1))


    # result_tensor = torch.vstack(result)
    # final_prob= torch.mean(result_tensor, dim=0)
    # final_prob_numpy = torch.mean(result_tensor, dim=0).cpu().detach().numpy()
    # top_3_idx = torch.argsort(final_prob, descending=True).cpu().detach().numpy()
    # top_3_idx = top_3_idx[:1]
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # probs_dict = {cls: float(p) for cls, p
    #             in zip(metadata['classes'], final_prob_numpy)}
    # ###Max
    # # result_tensor = torch.vstack(result)
    # # final_prob, _ = torch.max(result_tensor, dim=0)
    # # final_prob_numpy = final_prob.cpu().detach().numpy()
    # # top_3_idx = torch.argsort(final_prob, descending=True).cpu().detach().numpy()
    # # top_3_idx = top_3_idx[:1]
    # # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # res = ''
    # for idx in top_3_idx:
    #     res += '{}: {:.4f}\n'.format(metadata['classes'][idx], final_prob_numpy[idx])
    # final_conf = float(final_prob_numpy[top_3_idx[0]])
    # top_1 = top_3_idx[0] 
    # metrics = {
    #     "conf":      final_conf,
    #     "pred_cls":  metadata['classes'][top_1],  
    #     "probs":      probs_dict,
    #     "area_px":   frame.shape[0] * frame.shape[1],
    #     "mpp":       metadata.get("mpp", 0.25),
    #     "orig_img":  cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)
    # }
    # return frame, res, metrics

