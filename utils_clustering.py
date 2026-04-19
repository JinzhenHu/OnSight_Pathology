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


def reflect_pad_image(img, pad):
    if img.ndim == 3:
        return np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    else:
        return np.pad(img, ((pad, pad), (pad, pad)), mode="reflect")
    

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
    if 'lemon' not in _MODEL_CACHE:
        
        weight_path = hf_hub_download(repo_id="diamandislabii/Lemon", filename="lemon.pth.tar")
        stats_path = hf_hub_download(repo_id="diamandislabii/Lemon", filename="mean_std.json")
        
        transform = prepare_transform(stats_path, size=target_cell_size)
        weight_path_obj = Path(weight_path)
        model = get_vit_feature_extractor(weight_path_obj, model_name, img_size=target_cell_size)
        model.eval()
        model.to(device)
        
        _MODEL_CACHE['lemon'] = {
            'model': model,
            'transform': transform
        }
    
    cached_data = _MODEL_CACHE['lemon']
    return cached_data['model'], cached_data['transform'], device


def get_cached_kaiko(device):
    if 'kaiko' not in _MODEL_CACHE:
        model_name = "hf-hub:1aurent/vit_small_patch16_224.kaiko_ai_towards_large_pathology_fms"
        
        transform = v2.Compose([
            v2.ToImage(),
            v2.Resize(size=224, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(size=224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        model = timm.create_model(
            model_name,
            pretrained=True,
            #dynamic_img_size=True,
            num_classes=0 
        ).to(device).eval()
        
        
        _MODEL_CACHE['kaiko'] = {
            'model': model,
            'transform': transform
        }
    
    cached_data = _MODEL_CACHE['kaiko']
    return cached_data['model'], cached_data['transform'], device






@torch.inference_mode()
def extract_embeddings_lemon(crops, model, transform, device, batch_size=64):

    all_embs = []
    use_amp = "cuda" in str(device)

    for i in range(0, len(crops), batch_size):
        batch = crops[i : i + batch_size]
        
        batch_tensor = torch.stack([transform(img) for img in batch]).to(device)
        
        with torch.autocast(device_type=("cuda" if use_amp else "cpu"), enabled=use_amp, dtype=torch.float16):
            embs = model(batch_tensor)
            
        all_embs.append(embs.cpu().float().numpy())

    return np.concatenate(all_embs, axis=0)


@torch.inference_mode()
def extract_embeddings_kaiko(crops, model, transform, device, batch_size=64):

    all_embs = []
    use_amp = "cuda" in str(device)

    for i in range(0, len(crops), batch_size):
        batch = crops[i : i + batch_size]
        

        batch_tensor = torch.stack([transform(img) for img in batch]).to(device)
        
        with torch.autocast(device_type=("cuda" if use_amp else "cpu"), enabled=use_amp):
            embs = model(batch_tensor)

        all_embs.append(embs.cpu().float().numpy())

    return np.concatenate(all_embs, axis=0)



def cluster_embeddings_lemon(embeddings, k, random_state=0):
    n_samples = len(embeddings)
    

    if n_samples == 0:
        return np.array([], dtype=np.int32)

    valid_k = max(1, min(k, n_samples))
    if valid_k == 1:
        return np.zeros(n_samples, dtype=np.int32)

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
    

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)    
    
    if use_pca: 
        pca = PCA(n_components=0.95, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        df_out = pd.DataFrame(X_pca, columns=pca_columns, index=features_df.index)
    else:
        df_out = pd.DataFrame(X_scaled, columns=feature_cols, index=features_df.index)
        
    df_out["Label"] = features_df["Label"].values
    return df_out


def perform_keamns(df_ready_for_cluster, n_clusters):
    feature_cols = [col for col in df_ready_for_cluster.columns if col != "Label"]
    X = df_ready_for_cluster[feature_cols]
    
    n_samples = len(X)
    
    if n_samples == 0:
        return np.array([], dtype=np.int32)
        
    valid_k = max(1, min(n_clusters, n_samples))
    
    if valid_k == 1:
        return np.zeros(n_samples, dtype=np.int32)
    
    km = KMeans(n_clusters=valid_k, random_state=42, n_init=20)
    clusters = km.fit_predict(X)
    return clusters


def fast_cluster_overlay(img, masks, labels, clusters, n_clusters, alpha=0.6):
    """Vectorized PyQt-compatible mask overlay"""
    
    max_label = masks.max()
    cluster_lut = np.full(max_label + 1, -1, dtype=np.int32)
    cluster_lut[labels] = clusters
    cluster_map = cluster_lut[masks]

    # Create Color LUT
    color_lut = np.zeros((n_clusters + 1, 3), dtype=np.uint8)
    
    if n_clusters == 2:
        color_lut[0] = [255, 0, 0]  # Cluster 0: red
        color_lut[1] = [0, 255, 0]  # Cluster 1: green
    else:
        cmap = plt.get_cmap('gist_rainbow')
        for i in range(n_clusters):
            rgba = cmap(i / max(1, n_clusters - 1))
            color_lut[i] = [int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)]
    
    color_lut[-1] = [0, 0, 0]
    colored_mask = color_lut[cluster_map]
    
    # Fast Alpha Blending
    overlay = img.copy()
    mask_fg = cluster_map != -1
    overlay[mask_fg] = cv2.addWeighted(overlay[mask_fg], 1 - alpha, colored_mask[mask_fg], alpha, 0)
    
    return overlay


def custom_clustering(features_df, selected_features=None, use_pca=False):
    if selected_features and len(selected_features) > 0:
        feature_cols = [col for col in selected_features if col in features_df.columns]
    else:
        cols_to_exclude = ["Label"]
        feature_cols = [col for col in features_df.columns if col not in cols_to_exclude]
        
    X = features_df[feature_cols].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)    
    
    if use_pca: 
        n_comps = min(X_scaled.shape[1], 0.95 if X_scaled.shape[1] > 2 else X_scaled.shape[1])
        pca = PCA(n_components=n_comps, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        df_out = pd.DataFrame(X_pca, columns=pca_columns, index=features_df.index)
    else:
        df_out = pd.DataFrame(X_scaled, columns=feature_cols, index=features_df.index)
        
    df_out["Label"] = features_df["Label"].values
    return df_out

def clear_cluster_models_cache(keep=None):
    global _MODEL_CACHE
    
    if 'lemon' in _MODEL_CACHE and keep != 'lemon':
        #print("release Lemon cache...")
        del _MODEL_CACHE['lemon']['model']
        del _MODEL_CACHE['lemon']['transform']
        del _MODEL_CACHE['lemon']
        
    if 'kaiko' in _MODEL_CACHE and keep != 'kaiko':
        #print("release Kaiko cache...")
        del _MODEL_CACHE['kaiko']['model']
        del _MODEL_CACHE['kaiko']['transform']
        del _MODEL_CACHE['kaiko']
        
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()