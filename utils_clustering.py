import numpy as np
import cv2
import pandas as pd
import time
from sklearn.decomposition import PCA
import timm
import timm
import torch
import gc
from torchvision.transforms import v2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from skimage.measure import regionprops
from PIL import Image
from lemon.model import prepare_transform, get_vit_feature_extractor
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from pathlib import Path
_MODEL_CACHE = {}
def relabel_and_filter_masks(masks, min_area=20, max_area=100000, min_solidity=0.80):
    new_masks = np.zeros_like(masks, dtype=np.int32)
    new_id = 1
    for prop in regionprops(masks):
        area = prop.area
        solidity = prop.solidity if prop.solidity is not None else 0
        if area < min_area or area > max_area:
            continue
        if solidity < min_solidity:
            continue
        coords = prop.coords
        new_masks[coords[:, 0], coords[:, 1]] = new_id
        new_id += 1
    return new_masks

# =========================
# 1. 反射 padding，避免边缘 crop 出界
# =========================
def reflect_pad_image(img, pad):
    if img.ndim == 3:
        return np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    else:
        return np.pad(img, ((pad, pad), (pad, pad)), mode="reflect")
    
# =========================
# 2. 从每个 cell 生成带 context 的 crop
#    不建议只保留 cell mask；先保留周围组织上下文
# =========================
def build_cell_crops(
    img,
    masks,
    output_size=40,
    context_scale=1.5,
    min_side=40,
    max_side=96
):
    props = regionprops(masks)
    pad = max_side +10
    img_pad = reflect_pad_image(img, pad)

    rows = []
    crops = []

    for prop in props:
        r, c = prop.centroid
        eqd = prop.equivalent_diameter

        side = int(np.clip(context_scale * eqd, min_side, max_side))
        #print(f"Processing cell {prop.label}: centroid=({r:.1f}, {c:.1f}), eqd={eqd:.1f}, crop side={side}")
        half = side // 2

        rp = int(round(r)) + pad
        cp = int(round(c)) + pad

        crop = img_pad[rp-half:rp+half, cp-half:cp+half]
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            continue
        crop = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)
        crops.append(Image.fromarray(crop.astype(np.uint8)))

        rows.append({
            "label": prop.label,
            "centroid_row": float(r),
            "centroid_col": float(c),
            "area": float(prop.area), 
            "solidity": float(prop.solidity),
            "equivalent_diameter": float(eqd),
            "raw_crop_side": int(side),
        })

    df = pd.DataFrame(rows)
    return crops, df



def get_cached_lemon(device, model_name="vits8", target_cell_size=40):
    """加载自定义 ViT 模型和配套的 Transform (带缓存逻辑)"""
    
    # 1. 检查缓存中是否同时存在模型和 transform
    if 'lemon' not in _MODEL_CACHE:
        print("正在从 Hugging Face 下载并初始化模型...")
        
        # 下载权重和统计信息
        weight_path = hf_hub_download(repo_id="diamandislabii/Lemon", filename="lemon.pth.tar")
        stats_path = hf_hub_download(repo_id="diamandislabii/Lemon", filename="mean_std.json")
        
        # 初始化 Transform
        transform = prepare_transform(stats_path, size=target_cell_size)
        weight_path_obj = Path(weight_path)
        # 初始化模型
        model = get_vit_feature_extractor(weight_path_obj, model_name, img_size=target_cell_size)
        model.eval()
        model.to(device)
        
        # 存入缓存 (存成字典比较方便)
        _MODEL_CACHE['lemon'] = {
            'model': model,
            'transform': transform
        }
    
    # 2. 无论是否是第一次运行，都从缓存中取值并返回
    cached_data = _MODEL_CACHE['lemon']
    return cached_data['model'], cached_data['transform'], device


def get_cached_kaiko(device):
    """加载kaiko ViT 模型和配套的 Transform (带缓存逻辑)"""
    
    # 1. 检查缓存中是否同时存在模型和 transform
    if 'kaiko' not in _MODEL_CACHE:
        model_name = "hf-hub:1aurent/vit_small_patch16_224.kaiko_ai_towards_large_pathology_fms"
        
        # 初始化 Transform
        transform = v2.Compose([
            v2.ToImage(),
            v2.Resize(size=224, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(size=224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        # 使用 timm 加载
        model = timm.create_model(
            model_name,
            pretrained=True,
            #dynamic_img_size=True,
            num_classes=0 # num_classes=0 会自动移除分类头，直接输出特征(CLS Token)
        ).to(device).eval()
        
        
        # 存入缓存 (存成字典比较方便)
        _MODEL_CACHE['kaiko'] = {
            'model': model,
            'transform': transform
        }
    
    # 2. 无论是否是第一次运行，都从缓存中取值并返回
    cached_data = _MODEL_CACHE['kaiko']
    return cached_data['model'], cached_data['transform'], device






@torch.inference_mode()
def extract_embeddings_lemon(crops, model, transform, device, batch_size=64):
    """
    通用特征提取函数，使用自定义的 transform
    - crops: PIL 图像列表
    """
    all_embs = []
    use_amp = "cuda" in str(device)

    for i in range(0, len(crops), batch_size):
        batch = crops[i : i + batch_size]
        
        # 将 PIL 列表通过自定义 transform 转换为张量并推送到 GPU
        batch_tensor = torch.stack([transform(img) for img in batch]).to(device)
        
        # 使用自动混合精度 (AMP) 提速，注意指定 dtype=torch.float16
        with torch.autocast(device_type=("cuda" if use_amp else "cpu"), enabled=use_amp, dtype=torch.float16):
            embs = model(batch_tensor)
            
        # 转换为 float32 避免后续聚类(PCA/UMAP)的精度问题，并转为 numpy
        all_embs.append(embs.cpu().float().numpy())

    return np.concatenate(all_embs, axis=0)


@torch.inference_mode()
def extract_embeddings_kaiko(crops, model, transform, device, batch_size=64):
    """
    通用 Kaiko 特征提取函数
    - crops: PIL 图像列表
    """
    all_embs = []
    use_amp = "cuda" in str(device)

    for i in range(0, len(crops), batch_size):
        batch = crops[i : i + batch_size]
        
        # 将 PIL 列表转换为经过预处理的张量并推送到 GPU
        batch_tensor = torch.stack([transform(img) for img in batch]).to(device)
        
        # 💡 优化: 自动混合精度 (AMP)
        with torch.autocast(device_type=("cuda" if use_amp else "cpu"), enabled=use_amp):
            # timm 模型在 num_classes=0 时直接返回 (batch, features)
            embs = model(batch_tensor)
            
        # 转换为 float32 避免后续计算精度问题，并转为 numpy
        all_embs.append(embs.cpu().float().numpy())

    return np.concatenate(all_embs, axis=0)



def cluster_embeddings_lemon(embeddings, k, random_state=0):
    n_samples = len(embeddings)
    
    # 极盾防御 1：如果视野里压根没有细胞，直接返回空数组
    if n_samples == 0:
        return np.array([], dtype=np.int32)
        
    # 极盾防御 2：强制纠正 k 的值。k 不能小于 1，且不能大于细胞总数
    valid_k = max(1, min(k, n_samples))
    
    # 极盾防御 3：如果 k=0(被纠正为1) 或者只有 1 个细胞，不需要聚类，全部分配为 Cluster 0
    if valid_k == 1:
        return np.zeros(n_samples, dtype=np.int32)

    # 正常聚类流程
    scaler = StandardScaler()
    Xs = scaler.fit_transform(embeddings)
    
    n_features = Xs.shape[1]
    n_pcs = min(32, n_features, n_samples)
    
    if n_samples >= 2:
        pca = PCA(n_components=n_pcs, random_state=random_state)
        Xp = pca.fit_transform(Xs)
    else:
        Xp = Xs

    km = KMeans(n_clusters=valid_k, n_init=20, random_state=random_state)
    labels = km.fit_predict(Xp)
    return labels



def hand_crafted_clustering(features_df, use_pca = False):
    cols_to_exclude = ["Label"]
    feature_cols = [col for col in features_df.columns if col not in cols_to_exclude]
    X = features_df[feature_cols].fillna(0)
    

    # 1. 无论是否 PCA，聚类前【必须】标准化！
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)    
    
    # 2. 决定是否降维
    if use_pca: 
        pca = PCA(n_components=0.95, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        df_out = pd.DataFrame(X_pca, columns=pca_columns, index=features_df.index)
    else:
        # 不做 PCA 时，也要返回【标准化后】的特征给 K-Means
        df_out = pd.DataFrame(X_scaled, columns=feature_cols, index=features_df.index)
        
    # 将 Label 放回去，方便后续对应
    df_out["Label"] = features_df["Label"].values
    return df_out


def perform_keamns(df_ready_for_cluster, n_clusters):
    """只负责执行 KMeans，并加入防呆保护避免崩溃"""
    feature_cols = [col for col in df_ready_for_cluster.columns if col != "Label"]
    X = df_ready_for_cluster[feature_cols]
    
    n_samples = len(X)
    
    # 防呆 1：0 个细胞
    if n_samples == 0:
        return np.array([], dtype=np.int32)
        
    # 防呆 2：纠正非法的 K 值
    valid_k = max(1, min(n_clusters, n_samples))
    
    # 防呆 3：如果被强制为 1 个 cluster，跳过运算直接返回 0 标签
    if valid_k == 1:
        return np.zeros(n_samples, dtype=np.int32)
    
    km = KMeans(n_clusters=valid_k, random_state=42, n_init=20)
    clusters = km.fit_predict(X)
    return clusters

# def fast_cluster_overlay(img, masks, labels, clusters, n_clusters, alpha=0.6):
#     """Vectorized PyQt-compatible mask overlay (Replaces plt.show)"""
    
#     max_label = masks.max()
#     cluster_lut = np.full(max_label + 1, -1, dtype=np.int32)
#     cluster_lut[labels] = clusters
#     cluster_map = cluster_lut[masks]

#     # Create Color LUT
#     color_lut = np.zeros((n_clusters + 1, 3), dtype=np.uint8)
#     cmap = plt.get_cmap('gist_rainbow')
#     for i in range(n_clusters):
#         rgba = cmap(i / max(1, n_clusters - 1))
#         # Convert RGB to BGR for OpenCV
#         color_lut[i] = [int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)]
    
#     color_lut[-1] = [0, 0, 0] # Transparent background
#     colored_mask = color_lut[cluster_map]
    
#     # Fast Alpha Blending
#     overlay = img.copy()
#     mask_fg = cluster_map != -1
#     overlay[mask_fg] = cv2.addWeighted(overlay[mask_fg], 1 - alpha, colored_mask[mask_fg], alpha, 0)
    
#     return overlay
def fast_cluster_overlay(img, masks, labels, clusters, n_clusters, alpha=0.6):
    """Vectorized PyQt-compatible mask overlay (Replaces plt.show)"""
    
    max_label = masks.max()
    cluster_lut = np.full(max_label + 1, -1, dtype=np.int32)
    cluster_lut[labels] = clusters
    cluster_map = cluster_lut[masks]

    # Create Color LUT
    color_lut = np.zeros((n_clusters + 1, 3), dtype=np.uint8)
    
    # 🚀 核心修改：如果是 2 个 Cluster，强制使用红绿配色
    if n_clusters == 2:
        # 因为传入的 img 是 RGB 格式，所以颜色数组也按 [R, G, B] 排列
        color_lut[0] = [255, 0, 0]  # Cluster 0: 红色
        #color_lut[1] = [52, 152, 219]  # Cluster 1: 蓝色
        color_lut[1] = [0, 255, 0]  # Cluster 1: 绿色
    else:
        # 否则使用原本的 gist_rainbow 渐变色
        cmap = plt.get_cmap('gist_rainbow')
        for i in range(n_clusters):
            rgba = cmap(i / max(1, n_clusters - 1))
            color_lut[i] = [int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)]
    
    color_lut[-1] = [0, 0, 0] # Transparent background
    colored_mask = color_lut[cluster_map]
    
    # Fast Alpha Blending
    overlay = img.copy()
    mask_fg = cluster_map != -1
    overlay[mask_fg] = cv2.addWeighted(overlay[mask_fg], 1 - alpha, colored_mask[mask_fg], alpha, 0)
    
    return overlay


def custom_clustering(features_df, selected_features=None, use_pca=False):
    # 如果用户传了指定的特征列表，就只用这些列进行聚类
    if selected_features and len(selected_features) > 0:
        # 确保选中的列在 df 里真实存在，防呆
        feature_cols = [col for col in selected_features if col in features_df.columns]
    else:
        # 如果没传（后备方案），就用原来的逻辑排除 Label
        cols_to_exclude = ["Label"]
        feature_cols = [col for col in features_df.columns if col not in cols_to_exclude]
        
    X = features_df[feature_cols].fillna(0)
    
    # 1. 无论是否 PCA，聚类前【必须】标准化！
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)    
    
    # 2. 决定是否降维
    if use_pca: 
        # 防止特征数太少 PCA 报错
        n_comps = min(X_scaled.shape[1], 0.95 if X_scaled.shape[1] > 2 else X_scaled.shape[1])
        pca = PCA(n_components=n_comps, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        df_out = pd.DataFrame(X_pca, columns=pca_columns, index=features_df.index)
    else:
        df_out = pd.DataFrame(X_scaled, columns=feature_cols, index=features_df.index)
        
    # 将 Label 放回去，方便后续对应
    df_out["Label"] = features_df["Label"].values
    return df_out

# =========================
# 清理模型缓存
# =========================
def clear_cluster_models_cache(keep=None):
    """释放聚类模型显存，可以指定保留某一个"""
    global _MODEL_CACHE
    
    if 'lemon' in _MODEL_CACHE and keep != 'lemon':
        print("🧹 正在释放 Lemon 模型缓存...")
        del _MODEL_CACHE['lemon']['model']
        del _MODEL_CACHE['lemon']['transform']
        del _MODEL_CACHE['lemon']
        
    if 'kaiko' in _MODEL_CACHE and keep != 'kaiko':
        print("🧹 正在释放 Kaiko 模型缓存...")
        del _MODEL_CACHE['kaiko']['model']
        del _MODEL_CACHE['kaiko']['transform']
        del _MODEL_CACHE['kaiko']
        
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()