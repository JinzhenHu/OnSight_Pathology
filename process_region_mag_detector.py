from importlib import metadata

import mss
import torch
import numpy as np
import cv2
from torchvision.transforms import v2
import torch.nn as nn
import timm
import torch.nn.functional as F
from PIL import Image
from ctypes import wintypes
import ctypes
import re

def fix_region(region, tile_size):
    reg = region.copy()
    reg['width']  = max(reg['width'],  tile_size)
    reg['height'] = max(reg['height'], tile_size)
    return reg
# def get_foreground_window_scale():
#     user32 = ctypes.WinDLL("user32", use_last_error=True)
#     shcore = ctypes.WinDLL("Shcore", use_last_error=True)

#     PROCESS_PER_MONITOR_DPI_AWARE = 2
#     shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)

#     MDT_EFFECTIVE_DPI = 0
#     MONITOR_DEFAULTTONEAREST = 2

#     def get_scale_for_foreground_window():
#         hwnd = user32.GetForegroundWindow()
#         if not hwnd:
#             raise ctypes.WinError(ctypes.get_last_error())

#         hMonitor = user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
#         if not hMonitor:
#             raise ctypes.WinError(ctypes.get_last_error())

#         dpiX = wintypes.UINT()
#         dpiY = wintypes.UINT()

#         result = shcore.GetDpiForMonitor(
#             hMonitor,
#             MDT_EFFECTIVE_DPI,
#             ctypes.byref(dpiX),
#             ctypes.byref(dpiY)
#         )
#         if result != 0:
#             raise OSError(f"GetDpiForMonitor failed, HRESULT={result}")
            
            

#         scale_percent = dpiX.value / 96 * 100
#         return {
#             "hwnd": hwnd,
#             "dpi": dpiX.value,
#             "scale_percent": scale_percent
#         }

#     info = get_scale_for_foreground_window()
#     print(f"当前前台窗口所在显示器: DPI={info['dpi']}, 缩放={info['scale_percent']:.0f}%")
#     return info['scale_percent']/100
def get_foreground_window_scale():
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        shcore = ctypes.WinDLL("Shcore", use_last_error=True)

        PROCESS_PER_MONITOR_DPI_AWARE = 2
        # 尝试设置 DPI 感知，如果由于系统限制失败也不要紧
        try:
            shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)
        except Exception:
            pass

        MDT_EFFECTIVE_DPI = 0
        MONITOR_DEFAULTTONEAREST = 2

        # 1. 获取前台窗口
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            # 如果处于锁屏或切屏瞬间导致没有前台窗口，直接返回 1.0
            return 1.0

        # 2. 获取该窗口所在的显示器
        hMonitor = user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
        if not hMonitor:
            return 1.0

        dpiX = wintypes.UINT()
        dpiY = wintypes.UINT()

        # 3. 获取显示器 DPI
        result = shcore.GetDpiForMonitor(
            hMonitor,
            MDT_EFFECTIVE_DPI,
            ctypes.byref(dpiX),
            ctypes.byref(dpiY)
        )
        
        # 如果读取被拒绝 (比如遇到高权限窗口)，直接返回 1.0
        if result != 0:
            return 1.0

        scale_percent = dpiX.value / 96 * 100
        print(f"当前前台窗口所在显示器: DPI={dpiX.value}, 缩放={scale_percent:.0f}%")
        return scale_percent / 100.0

    except Exception as e:
        # 终极兜底：无论发生什么奇怪的 Windows API 报错，都不要崩溃，按 1.0 算
        print(f"[Warning] 无法读取系统缩放比例 ({e})，回退至默认 100% 缩放。")
        return 1.0
    


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
    preprocessing = v2.Compose([
        v2.ToImage(),
        #v2.Resize(size=224),
        v2.CenterCrop(size=224),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)),
    ])

    metadata = kwargs['metadata']
    model = kwargs['model']
    tile_size = metadata['tile_size']
    os_scale = get_foreground_window_scale()

    with mss.mss() as sct:
        # 例如 tile_size 是 224，os_scale 是 2.0，我们实际需要从屏幕截取至少 448x448 的区域。
        #required_min_size = int(tile_size * os_scale)
        region = fix_region(region, tile_size)
        screenshot = sct.grab(region)



    frame_orig = np.array(screenshot, dtype=np.uint8)
    #print(frame.shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = cv2.cvtColor(frame_orig, cv2.COLOR_BGRA2RGB)
    if abs(os_scale - 1.0) > 1e-3:
        print(f"检测到屏幕缩放: {os_scale:.2f}x，正在调整输入图像大小以适配模型...")
        frame = cv2.resize(
            frame,
            (max(1, int(round(frame.shape[1] / os_scale))),
            max(1, int(round(frame.shape[0] / os_scale)))),
            interpolation=cv2.INTER_AREA
        )
    else: 
        print("屏幕缩放为 100%，无需调整输入图像大小。")
    h, w = frame.shape[:2]

    tile_size = metadata['tile_size']         # 1024
    stride     = tile_size               # 可调：无重叠 = tile_size
    tile_slices = generate_tiles_nopad(h, w, tile_size, stride)

    result = []
    model.eval()


    # # >>> [ATTENTION MAP START] >>>
    # attention_maps = {}
    # def get_attention(module, input, output):
    #     attention_maps['attn'] = output.detach().cpu()

    # # Handle PyTorch 2.0+ Flash Attention compatibility
    # original_fused_attn = getattr(model.blocks[-1].attn, 'fused_attn', None)
    # if original_fused_attn is not None:
    #     model.blocks[-1].attn.fused_attn = False

    # handle = model.blocks[-1].attn.attn_drop.register_forward_hook(get_attention)

    # # Empty arrays to stitch the attention map back to the original frame size
    # full_attn = np.zeros((h, w), dtype=np.float32)
    # weight_map = np.zeros((h, w), dtype=np.float32)
    # # <<< [ATTENTION MAP END] <<<
# +++ NEW +++
    # 使用列表存储所有层的 qkv，并注册多个 hook
    # attention_maps = []
    # def get_qkv(module, input, output):
    #     attention_maps.append(output.detach().cpu())

    # hooks = []
    # for block in model.blocks:
    #     hook = block.attn.qkv.register_forward_hook(get_qkv)
    #     hooks.append(hook)
    # # <<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # # Empty arrays to stitch the attention map back to the original frame size
    # full_attn = np.zeros((h, w), dtype=np.float32)
    # weight_map = np.zeros((h, w), dtype=np.float32)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # for rs, cs in tile_slices:
    #     tile_img = Image.fromarray(frame[rs, cs])
    #     tile_tensor = preprocessing(tile_img).unsqueeze(0).to(device)
    #     with torch.no_grad():
    #         out = model(tile_tensor)
    #         result.append(F.softmax(out, dim=1))

    #         # >>> [ATTENTION MAP START] >>>
    #         if 'attn' in attention_maps:
    #             attn = attention_maps['attn']
    #             # Process the CLS token's attention to the spatial patches (14x14)
    #             cls_attn = attn[0, :, 0, 1:].mean(dim=0).reshape(14, 14).unsqueeze(0).unsqueeze(0)
                
    #             # Dynamically resize the 14x14 map to match this specific tile's size
    #             tile_h = rs.stop - rs.start
    #             tile_w = cs.stop - cs.start
    #             cls_attn = F.interpolate(cls_attn, size=(tile_h, tile_w), mode='bilinear', align_corners=False).squeeze().numpy()
                
    #             # Add to our full frame stitcher
    #             full_attn[rs, cs] += cls_attn
    #             weight_map[rs, cs] += 1.0
    #         # <<< [ATTENTION MAP END] <<<

    # # >>> [ATTENTION MAP START] >>>
    # handle.remove() # Clean up hook
    # if original_fused_attn is not None:
    #     model.blocks[-1].attn.fused_attn = original_fused_attn

    # # Average out overlapping regions and normalize between 0.0 and 1.0
    # full_attn = full_attn / (weight_map + 1e-8)
    # if full_attn.max() > 0:
    #     full_attn = (full_attn - full_attn.min()) / (full_attn.max() - full_attn.min() + 1e-8)
    # # <<< [ATTENTION MAP END] <<<
    for rs, cs in tile_slices:
        tile_img = Image.fromarray(frame[rs, cs])
        tile_tensor = preprocessing(tile_img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tile_tensor)
            result.append(F.softmax(out, dim=1))

            # >>> [ATTENTION MAP START] >>>
            # --- OLD ---
            # if 'attn' in attention_maps:
            #     attn = attention_maps['attn']
            #     cls_attn = attn[0, :, 0, 1:].mean(dim=0).reshape(14, 14).unsqueeze(0).unsqueeze(0)
            
#             # +++ NEW +++
#             if len(attention_maps) > 0:
#                 # 提取并计算每一层的 Attention 矩阵
#                 attentions = []
#                 for qkv in attention_maps:
#                     B, N, C_times_3 = qkv.shape
#                     C = C_times_3 // 3
#                     num_heads = model.blocks[0].attn.num_heads
#                     head_dim = C // num_heads

#                     # 分离并计算注意力: Q @ K^T
#                     qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
#                     q, k, v = qkv[0], qkv[1], qkv[2]
#                     scale = head_dim ** -0.5
#                     attn = (q @ k.transpose(-2, -1)) * scale
#                     attn = attn.softmax(dim=-1).mean(dim=1) # 在 head 维度取平均 -> [B, N, N]
#                     attentions.append(attn[0]) # 取当前 batch 存入 -> [N, N]

#                 # Rollout 连乘算法
#                 N = attentions[0].shape[0]
#                 result_attn = torch.eye(N)
#                 for a in attentions:
#                     a = a + torch.eye(N) # 残差连接补全
#                     a = a / a.sum(dim=-1, keepdim=True) # 重新归一化
#                     result_attn = torch.matmul(a, result_attn)

#                 # 提取 CLS 对其余 Patch 的注意力流
#                 cls_attn = result_attn[0, 1:]
#                 num_patch = int(cls_attn.shape[0] ** 0.5) # 通常是 14
#                 cls_attn = cls_attn.reshape(1, 1, num_patch, num_patch) # 变为 [1, 1, 14, 14]

#                 # --- 极其重要的一步！清空当前 tile 积累的 maps，供下一个 tile 使用 ---
#                 attention_maps.clear() 
#             # <<<<<<<<<<<<<<<<<<<<<<<<<<<<

#                 # Dynamically resize the map to match this specific tile's size
#                 tile_h = rs.stop - rs.start
#                 tile_w = cs.stop - cs.start
#                 cls_attn_np = F.interpolate(cls_attn, size=(tile_h, tile_w), mode='bilinear', align_corners=False).squeeze().numpy()
                
#                 # Add to our full frame stitcher
#                 full_attn[rs, cs] += cls_attn_np
#                 weight_map[rs, cs] += 1.0
#             # <<< [ATTENTION MAP END] <<<
# # +++ NEW +++
#     # 移除挂在所有层上的 hook
#     for hook in hooks:
#         hook.remove()
#     # <<<<<<<<<<<<<<<<<<<<<<<<<<<<

#     # Average out overlapping regions and normalize between 0.0 and 1.0
#     full_attn = full_attn / (weight_map + 1e-8)
#     if full_attn.max() > 0:
#         full_attn = (full_attn - full_attn.min()) / (full_attn.max() - full_attn.min() + 1e-8)
#     # <<< [ATTENTION MAP END] <<<
#     # >>> [ATTENTION MAP START] >>>
#     # --- OLD ---
#     # handle.remove()
#     # if original_fused_attn is not None:
#     #     model.blocks[-1].attn.fused_attn = original_fused_attn

#     # +++ NEW +++
#     # 移除挂在所有层上的 hook
#     for hook in hooks:
#         hook.remove()
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # Average out overlapping regions and normalize between 0.0 and 1.0
    # full_attn = full_attn / (weight_map + 1e-8)
    # if full_attn.max() > 0:
    #     full_attn = (full_attn - full_attn.min()) / (full_attn.max() - full_attn.min() + 1e-8)
    # <<< [ATTENTION MAP END] <<<
# ---------------------------------------------------------
    # 最简单粗暴的方式：直接写死倍数，绝对不报错！
    # ---------------------------------------------------------
    result_tensor = torch.vstack(result)
    final_prob = torch.mean(result_tensor, dim=0)
    final_prob_numpy = final_prob.cpu().detach().numpy()

    # 1. 硬编码你的 4 个倍数 (5x, 10x, 20x, 40x)
    hardcoded_mags = np.array([5.0, 10.0, 20.0, 40.0])
    
    # 2. 直接概率相乘求和
    continuous_mag = np.sum(final_prob_numpy * hardcoded_mags)

    # 3. 拿到 Top 1 结果
    top_3_idx = torch.argsort(final_prob, descending=True).cpu().detach().numpy()
    top_1 = top_3_idx[0]
    final_conf = float(final_prob_numpy[top_1])

    # 4. 最简单的文本拼接
    res = f"Predicted Magnification: {metadata['classes'][top_1]}<br>"
    res += f"Continuous Magnification: {continuous_mag:.2f}X"

    # ---------------------------------------------------------
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    probs_dict = {cls: float(p) for cls, p in zip(metadata['classes'], final_prob_numpy)}
    print(f"Predicted Magnification: {metadata['classes'][top_1]}")
    
    metrics = {
        "conf":          float(continuous_mag),
        "pred_cls":      metadata['classes'][top_1],  
        "continuous_mag": float(continuous_mag),
        "probs":         probs_dict,
        "area_px":       frame.shape[0] * frame.shape[1],
        "mpp":           metadata.get("mpp", 0.25),
        "orig_img":      cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB),
        #"attention_map": full_attn 
    }
    
    return frame, res, metrics
    # for rs, cs in tile_slices:
    #     tile_img = Image.fromarray(frame[rs, cs])
    #     tile_tensor = preprocessing(tile_img).unsqueeze(0).to(device)
    #     with torch.no_grad():
    #         out = model(tile_tensor)
    #         result.append(F.softmax(out, dim=1))



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

