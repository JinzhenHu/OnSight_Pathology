from importlib import metadata

import mss
import numpy as np
import cv2
from skimage.color import rgb2hsv
from skimage.measure import label, regionprops_table
import numpy as np
import numpy as np

from skimage import io, color, filters
from skimage import morphology

from vendor.cp_core_4_2_8.cellprofiler_core.constants import image
# 之前的辅助函数保持不变 (_safe_float, fix_region, visualize_overlay, get_tissue_mask_global)
# get_tile_coordinates 已经不再需要了，可以保留也可以删除

def _safe_float(value, default=0.2):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def fix_region(region, tile_size):
    reg = region.copy()
    # 即使不切片，保持一定的尺寸倍数通常对卷积神经网络也是好的（虽然这里主要为了防止报错）
    reg['width'] = max(reg['width'], tile_size)
    reg['height'] = max(reg['height'], tile_size)
    return reg

def visualize_overlay(img, mask, color=(0, 255, 0), alpha=0.5):
    if alpha <= 0:
        return img
    overlay = img.copy()
    cell_region = (mask > 0)
    overlay[cell_region] = color
    blended = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    return blended

def get_tissue_mask_global(img_rgb):
    hsv = rgb2hsv(img_rgb)

    saturation = hsv[:, :, 1]

    tissue_mask = saturation > 0.05
    # gray_image = color.rgb2gray(img_rgb)
    # thresh = filters.threshold_otsu(gray_image)
    # tissue_mask = gray_image < thresh*1.3
    # tissue_mask = morphology.remove_small_objects(tissue_mask, max_size=500)  # 移除小于 500 像素的区域
    # tissue_mask = morphology.remove_small_holes(tissue_mask,max_size=500 )  # 填充小于 500 像素的孔洞
    return tissue_mask

def process_region(region, **kwargs):
    metadata = kwargs['metadata']
    model = kwargs['model']
    configs = kwargs.get('additional_configs', {})
    
    # tile_size 在这里只用于 fix_region 确保截图不为空或过小，不再用于切片
    tile_size = metadata.get('tile_size', 256) 
    show_overlay = configs.get('show_overlay', True)
    transparency = _safe_float(configs.get('mask_transparency', 0.5))
    _overlay_alpha = max(0.0, min(1.0, 1.0 - transparency))

    # 1. 截图
    with mss.mss() as sct:
        region = fix_region(region, tile_size)
        screenshot = sct.grab(region)

    frame_orig = np.array(screenshot, dtype=np.uint8)
    frame = cv2.cvtColor(frame_orig, cv2.COLOR_BGRA2RGB)
    h, w = frame.shape[:2]

    # 2. 模型推理 (直接处理整张图，不再切片)
    # 警告：如果 region 非常大，这里可能会显存溢出
    labeled_output, _,_ = model.eval(frame)
    
    # # 3. 处理输出结果
    # # 将 Tensor 转为 numpy，并去掉多余维度
    # current_mask = labeled_output
    
    # # 生成 bool 类型的 mask
    # full_nuclei_mask = current_mask > 0

    # # 4. 后处理与指标计算 (逻辑保持不变)
    # labeled_full_mask = label(full_nuclei_mask)
    total_cells_count = labeled_output.max()

    # 初始化 4 个指标
    ecc_mean = 0.0
    ecc_median = 0.0
    ecc_mid50_mean = 0.0
    ecc_std = 0.0

    if total_cells_count > 0:
        props = regionprops_table(labeled_output, properties=['eccentricity'])
        ecc_values = props['eccentricity']
        
        if len(ecc_values) > 0:
            # 1. 全部均值 (Mean)
            ecc_mean = np.mean(ecc_values)
            # 2. 中位数 (Median)
            ecc_median = np.median(ecc_values)
            # 3. 标准差 (Standard Deviation)
            ecc_std = np.std(ecc_values)
            
            # 4. 中间 50% 的均值
            q1 = np.percentile(ecc_values, 25)
            q3 = np.percentile(ecc_values, 75)
            middle_50_ecc = ecc_values[(ecc_values >= q1) & (ecc_values <= q3)]
            if len(middle_50_ecc) > 0:
                ecc_mid50_mean = np.mean(middle_50_ecc)

    # Cellularity
    tissue_mask_full = get_tissue_mask_global(frame)
    #print(np.sum(tissue_mask_full)/(tissue_mask_full.shape[0]*tissue_mask_full.shape[1]))
    tissue_area_px = np.sum(tissue_mask_full)

    valid_nuclei_mask = (labeled_output > 0) & tissue_mask_full

    nuclei_area_px = np.sum(valid_nuclei_mask)

    cellularity_score = nuclei_area_px / tissue_area_px
    
    if tissue_area_px > 0:
        cellularity_score = nuclei_area_px / tissue_area_px
    else:
        cellularity_score = 0.0
        
    if show_overlay:
        final_vis_img = visualize_overlay(frame, labeled_output, color=(0, 255, 0), alpha=_overlay_alpha)
    else:
        final_vis_img = frame

    final_vis_img_bgr = cv2.cvtColor(final_vis_img, cv2.COLOR_RGB2BGR)

    # 文本结果展示更新
    text = ""
    text += f"Cellularity: {cellularity_score * 100:.2f} %\n"
    text += f"Ecc Mean: {ecc_mean:.4f} | Median: {ecc_median:.4f}\n"
    text += f"Ecc Mid-50% Mean: {ecc_mid50_mean:.4f} | Std: {ecc_std:.4f}\n"

    # 将 4 个指标存入 metrics 传给 GUI
    metrics = {
        "cellularity": float(cellularity_score),
        "ecc_mean": float(ecc_mean),
        "ecc_median": float(ecc_median),
        "ecc_mid50_mean": float(ecc_mid50_mean),
        "ecc_std": float(ecc_std),
        "orig_img": cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB),
        "area_px": frame.shape[0] * frame.shape[1],
        "mpp": metadata.get("mpp", 0.25),
    }
    
    return final_vis_img_bgr.astype(np.uint8), text, metrics