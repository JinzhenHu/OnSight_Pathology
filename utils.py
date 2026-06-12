import os
import sys
import timm
import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
import numpy as np
import histomicstk as htk
from transformers import AutoModel
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from CellViT.models.segmentation.cell_segmentation.cellvit import CellViTSAM
from CellViT.models.segmentation.cell_segmentation.cellvit_shared import CellViTSAMShared
from device_compat import get_device, is_gpu_available
_KMEANS_KW = {'algorithm': 'lloyd'} if sys.platform == 'darwin' else {}

# Always use this for accessing any local path
def resource_path(relative_path):
    """Get absolute path to resource (for dev and for PyInstaller onefile mode)"""
    if hasattr(sys, '_MEIPASS'):
        # _MEIPASS is the temp folder where PyInstaller unpacks files
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

import logging

# ============================================================================
# Build mode detection (runs once at import)
# ----------------------------------------------------------------------------
# LOCAL+HF mode: bundled_models/ directory exists → prefer bundled weights,
#                fall back to HuggingFace download if a specific file missing.
# HF-ONLY mode:  no bundled_models/ → always download from HuggingFace.
#
# The packaging script (app.spec) decides which mode to build by including
# or excluding the bundled_models/ folder. Runtime requires no changes.
# ============================================================================
_BUNDLED_DIR = resource_path("bundled_models")
BUNDLE_MODE_ENABLED = os.path.isdir(_BUNDLED_DIR)
logging.info(
    f"[build_mode] {'LOCAL+HF' if BUNDLE_MODE_ENABLED else 'HF-ONLY'} "
    f"(bundled_models dir: {_BUNDLED_DIR})"
)

def get_bundled_model_names():
    """Return list of human-friendly names of bundled models (for UI display)."""
    if not BUNDLE_MODE_ENABLED:
        return []
    try:
        import settings
        names = []
        for cat, items in settings.MODEL_CATELOG:
            for item in items:
                meta = settings.MODEL_METADATA.get(item['name'], {})
                local_rel = meta.get("local_path")
                if local_rel and os.path.exists(resource_path(local_rel)):
                    names.append(item['name'])
        return names
    except Exception:
        return []

def _get_weights_path(model_info, hf_repo, hf_filename):
    """
    Resolves weights path. Order:
      1. Bundled local file (if exists)
      2. Download from HF to an app-owned directory using `local_dir`
    """
    # ----- 1. Try bundled -----
    if BUNDLE_MODE_ENABLED:
        local_rel = model_info.get("local_path")
        if local_rel:
            local_abs = resource_path(local_rel)
            if os.path.exists(local_abs):
                logging.info(f"[weights] Loading bundled: {local_abs}")
                return local_abs

    # ----- 2. Download to local_dir (REAL FILES, no symlinks) -----
    from huggingface_hub import hf_hub_download

    # Per-repo subdirectory under our app-owned cache
    safe_repo = hf_repo.replace("/", "__")
    if sys.platform == "darwin":
        _default_cache = os.path.join(os.path.expanduser("~"), "Library", "Application Support", "OnSightPathology", "hf_cache", "hub")
    else:
        _default_cache = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "OnSightPathology", "hf_cache", "hub")

    download_dir = os.path.join(
        os.environ.get("HF_HUB_CACHE", _default_cache),
        "_local_dir_downloads",
        safe_repo,
    )
    os.makedirs(download_dir, exist_ok=True)

    target = os.path.join(download_dir, hf_filename)

    # If we already downloaded once and the file is non-empty, reuse it
    if os.path.exists(target) and os.path.getsize(target) > 0:
        logging.info(f"[weights] Reusing existing download: {target}")
        return target

    logging.info(f"[weights] Downloading to local_dir: {download_dir}")
    path = hf_hub_download(
        repo_id=hf_repo,
        filename=hf_filename,
        local_dir=download_dir,    
    )
    return path

def extract_tiles(frame, tile_size):
    slices = []

    # capture_size > tile_size: split capture window into sub tiles
    # capture_size < tile_size: no splitting. use the capture window dimension.
    # - there is a message on gui telling user that choosing a capture size less than the trained tile size is not recommended...

    tile_size_y = tile_size_x = tile_size

    if frame.shape[0] < tile_size:
        tile_size_y = frame.shape[0]
    if frame.shape[1] < tile_size:
        tile_size_x = frame.shape[1]

    for i in range(frame.shape[0] // tile_size_y):
        for j in range(frame.shape[1] // tile_size_x):
            tile_slice = frame[
                         (i * tile_size_y):((i * tile_size_y) + tile_size_y),
                         (j * tile_size_x):((j * tile_size_x) + tile_size_x),
                         :
                         ]
            slices.append(tile_slice)

    return slices


class MidnightClassifier(nn.Module):
    def __init__(self, model_name="kaiko-ai/midnight", num_classes=3, dropout=0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size  

        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, num_classes))

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        tokens = outputs.last_hidden_state        
        cls_token = tokens[:, 0, :]               
        patch_mean = tokens[:, 1:, :].mean(dim=1) 
        feat = torch.cat([cls_token, patch_mean], dim=1)  
        logits = self.head(feat)
        return logits


    
def load_model(model_info):
    res = {
        'model': None,
        'process_region_func': None,
        'using_gpu': False
    }
    ##############################################################################
    # ONNX
    ##############################################################################
    if model_info['repo_src'] == 'HuggingFace':
        ##############################################################################
        # VIT-Tumor-4-kaiko
        ##############################################################################
        if model_info['model'] == 'VITtumor_kaiko':
            from huggingface_hub import hf_hub_download
            model_path = _get_weights_path(model_info, model_info['repo'], "best_kaikuo_20000image.pth")
            #model_path = r"D:\UofT\2025fall\OnSight\Revisions\4_Class_Tumor_VIT\best_model_tumor_4class.pth"
            import timm
            import torch
            import torch.nn as nn
            from process_region_VIT_tumor4_kaiko import process_region
            # Create Model
            model = timm.create_model(
                #model_name="hf-hub:1aurent/vit_base_patch16_224.kaiko_ai_towards_large_pathology_fms",
                model_name="vit_base_patch16_224",
                dynamic_img_size=True,
                pretrained=False, )

            device = get_device()
            num_features = model.num_features
            num_classes = 4
            # model.head = nn.Sequential(
            #     nn.LayerNorm(num_features),  
            #     nn.Linear(num_features, 256),  
            #     nn.GELU(),                  
            #     nn.Dropout(0.5),              
            #     nn.Linear(256, num_classes)    
            # )
            model.head = nn.Sequential(
                nn.Linear(num_features, num_classes),
            )
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            res['model'] = model
            res['process_region_func'] = process_region
            res['using_gpu'] = is_gpu_available()
            return res
        ##############################################################################
        # VIT-Magnification-4-kaiko
        ##############################################################################
        if model_info['model'] == 'VITmagnification_kaiko':
            from huggingface_hub import hf_hub_download
            #model_path = r"D:\UofT\2025fall\OnSight\Revisions\Mag_Detector\best_model.pth"
            model_path = _get_weights_path(model_info, "diamandislabii/Magnification_Detection", "best_model.pth")
            import timm
            import torch
            import torch.nn as nn
            from process_region_mag_detector import process_region
            # Create Model
            model = timm.create_model(
            #model_name="hf-hub:1aurent/vit_small_patch16_224.kaiko_ai_towards_large_pathology_fms",
            model_name="vit_small_patch16_224",
               # dynamic_img_size=True,
            pretrained=False, )


            device = get_device()
            num_features = model.num_features
            num_classes = 4
            model.head = nn.Sequential(
                nn.Linear(num_features, 1024),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(1024, num_classes)
            )
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            res['model'] = model
            res['process_region_func'] = process_region
            res['using_gpu'] = is_gpu_available()
            return res
       

        ##############################################################################
        # CellViT SAM-H
        ##############################################################################
        if model_info['model'] == 'CellViT':
            import torch
            from process_region_cellvit import process_region
            from huggingface_hub import hf_hub_download

            def _unflatten_dict(flat_dict, sep="."):
                result = {}
                for key, value in flat_dict.items():
                    parts = key.split(sep)
                    d = result
                    for p in parts[:-1]:
                        if p not in d:
                            d[p] = {}
                        d = d[p]
                    d[parts[-1]] = value
                return result

            def _clean_state_dict(state_dict):
                cleaned = {}
                for k, v in state_dict.items():
                    nk = k
                    if nk.startswith("module."):
                        nk = nk[len("module."):]
                    if nk.startswith("model."):
                        nk = nk[len("model."):]
                    cleaned[nk] = v
                return cleaned

            def load_cellvit_sam_weights(ckpt_path, device):
                ckpt = torch.load(ckpt_path, map_location="cpu")

                run_conf_flat = ckpt["config"]
                run_conf = _unflatten_dict(run_conf_flat, ".")
                arch = ckpt["arch"]

                #print("CellViT checkpoint arch:", arch)
                #print("CellViT backbone:", run_conf["model"].get("backbone", None))


                model = CellViTSAM(
                    model_path=None,
                    num_nuclei_classes=run_conf["data"]["num_nuclei_classes"],
                    num_tissue_classes=run_conf["data"]["num_tissue_classes"],
                    vit_structure=run_conf["model"]["backbone"],
                    regression_loss=run_conf["model"].get("regression_loss", False),
                )

                state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", None))
                msg = model.load_state_dict(_clean_state_dict(state_dict), strict=False)
                #print("CellViT SAM load_state_dict:", msg)

                model.eval()
                model.to(device)

                mean = run_conf_flat.get("transformations.normalize.mean", [0.5, 0.5, 0.5])
                std = run_conf_flat.get("transformations.normalize.std", [0.5, 0.5, 0.5])

                nuclei_types = {}
                tissue_types = {}

                for k, v in run_conf_flat.items():
                    if k.startswith("dataset_config.nuclei_types."):
                        name = k.replace("dataset_config.nuclei_types.", "")
                        nuclei_types[name] = int(v)

                    if k.startswith("dataset_config.tissue_types."):
                        name = k.replace("dataset_config.tissue_types.", "")
                        tissue_types[name] = int(v)


                nuclei_types = {
                    "Background": 0,
                    "Neoplastic": 1,
                    "Inflammatory": 2,
                    "Connective": 3,
                    "Dead": 4,
                    "Epithelial": 5,
                }

                type_id_to_name = {v: k for k, v in nuclei_types.items()}
                tissue_id_to_name = {v: k for k, v in tissue_types.items()} if tissue_types else None

                return model, mean, std, type_id_to_name, tissue_id_to_name, device

            device = get_device()

            model_path = _get_weights_path(model_info, model_info['repo'], "CellViT-SAM-H-x40.pth")

            model_file, mean, std, type_id_to_name, tissue_id_to_name, device = load_cellvit_sam_weights(
                model_path,
                device,
            )

            model = [
                model_file,
                mean,
                std,
                type_id_to_name,
                tissue_id_to_name,
                device,
            ]

            res['model'] = model
            res['process_region_func'] = process_region
            res['using_gpu'] = is_gpu_available()
            return res

        ##############################################################################
        # Retinanet
        ##############################################################################
        if model_info['model'] == 'Retinanet':
            import torch
            import yaml
            from retinanet.Model import MyMitosisDetection
            from huggingface_hub import hf_hub_download

            device = get_device()

            model_path = _get_weights_path(model_info, model_info['repo'], "bestmodel.pth")
            config_path = resource_path("retinanet/file/config.yaml")

            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            def _safe_float(value, default=0.2):
                """
                Return float(value) if it's a valid numeric string/number,
                otherwise return the default.
                """
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return default

            thred = 0.1
            Cl = _safe_float(model_info['additional_configs'].get('specificity', 0))

            detector = MyMitosisDetection(model_path, config, Cl, thred)
            model = detector.load_model()
            from process_region_mitosis import process_region
            res['model'] = detector
            res['process_region_func'] = process_region
            res['using_gpu'] = is_gpu_available()

            return res

        ##############################################################################
        # Cellpose-SAM
        ##############################################################################
        if model_info['model'] == 'Cellpose-SAM':
            from cellpose import models
            import torch
            from process_region_cellpose import process_region

            res['model'] = models.CellposeModel(gpu=is_gpu_available())
            res['process_region_func'] = process_region
            res['using_gpu'] = is_gpu_available()

            return res

        ##############################################################################
        # YOLO
        ##############################################################################
        if model_info['model'] == 'YOLO':
            from ultralytics import YOLO
            import torch
            from process_region_YOLO import process_region
            from huggingface_hub import hf_hub_download

            weights_path = _get_weights_path(model_info, model_info['repo'], "best.pt")

            res['model'] = YOLO(weights_path)
            res['process_region_func'] = process_region
            res['using_gpu'] = is_gpu_available()

            return res
        
    ##############################################################################
    # cellprofiler
    ##############################################################################

    if model_info['model'] == 'CPSAM_profiler':
        from cellpose import models
        import torch
        from process_region_cpsam_cellfeatures import process_region

        device = str(get_device())
        model = models.CellposeModel(gpu=is_gpu_available())
        
        res['model'] = model
        res['process_region_func'] = process_region
        res['using_gpu'] = is_gpu_available()
        return res

    return res


def get_gpu_memory():
    """Return total GPU memory (in GB) if CUDA is available, else None."""
    import torch

    if not torch.cuda.is_available():
        return None
    device = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    return total


def get_system_memory():
    """Return total system RAM (in GB)."""
    import psutil

    return psutil.virtual_memory().total / (1024 ** 3)


def build_precision_labels(gpu_mem_gb, cpu_ram_gb):
    """Return a dict of precision → display label with GPU/CPU availability notes."""

    PRECISION_DISPLAY_MAP = {
        "4bit": "Fastest Speed",
        "8bit": "Balanced",
        "16bit": "Highest Quality"
    }

    labels = {}
    required = {
        "4bit": 9.1,
        "8bit": 11.2,
        "16bit": 16.7,
    }

    for k, base_label in PRECISION_DISPLAY_MAP.items():
        if gpu_mem_gb is None:
            # No GPU available → CPU fallback only
            if cpu_ram_gb >= 32:  # heuristic cutoff
                labels[k] = f"{base_label} — CPU fallback (minutes per token)"
            else:
                labels[k] = f"{base_label} — CPU fallback may fail (low RAM)"
        else:
            req = required[k]
            if gpu_mem_gb >= req:
                labels[k] = f"{base_label}"  # fully supported
            elif gpu_mem_gb >= req - 2:  # within ~2 GB tolerance
                labels[k] = f"{base_label} — May load but unstable/slow"
            else:
                labels[k] = f"{base_label} — Too large for your GPU"

    return labels

##############################################################################
# ROI_Finder Hotspot
##############################################################################
def normalize_to_01(arr):
    arr = arr.astype(np.float32)
    mn, mx = np.min(arr), np.max(arr)
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)

def get_tissue_mask(im_rgb, min_tissue_area=500, kernel_size=5):
    gray = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2GRAY)

    _, tissue_mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    tissue_mask = tissue_mask.astype(np.uint8)


    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(tissue_mask, 8)
    cleaned = np.zeros_like(tissue_mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_tissue_area:
            cleaned[labels == i] = 255

    return (cleaned > 0).astype(np.uint8)


def get_he_deconvolution(im_rgb):

    stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
    W = np.array([
        stain_color_map['hematoxylin'],
        stain_color_map['eosin'],
        stain_color_map['null']
    ]).T

    deconv = htk.preprocessing.color_deconvolution.color_deconvolution(im_rgb, W)

    imH = deconv.Stains[:, :, 0].astype(np.float32)
    imE = deconv.Stains[:, :, 1].astype(np.float32)

    return deconv, imH, imE




def compute_suspicious_score_map(
    im_rgb,
    imH,
    imE,
    tissue_mask,
    sigma_region=24
):
    tissue_float = tissue_mask.astype(np.float32)

    H_strength = (255.0 - imH) * tissue_float
    E_strength = (255.0 - imE) * tissue_float

    R = im_rgb[:, :, 0].astype(np.float32) * tissue_float
    G = im_rgb[:, :, 1].astype(np.float32) * tissue_float
    B = im_rgb[:, :, 2].astype(np.float32) * tissue_float


    Hn = normalize_to_01(H_strength)
    En = normalize_to_01(E_strength)
    #Tn = normalize_to_01(texture_map)

    H_region = cv2.GaussianBlur(Hn, (0, 0), sigmaX=sigma_region, sigmaY=sigma_region)
    E_region = cv2.GaussianBlur(En, (0, 0), sigmaX=sigma_region, sigmaY=sigma_region)
    # T_region = cv2.GaussianBlur(Tn, (0, 0), sigmaX=sigma_region, sigmaY=sigma_region)

    red_penalty = np.clip(E_region - H_region, 0, 1)
    local_tissue = cv2.GaussianBlur(tissue_float, (0, 0), sigmaX=sigma_region, sigmaY=sigma_region)

    score = (
       1 * H_region -
       2 * red_penalty
    )

    score = score / (local_tissue + 1e-6)
    score[local_tissue < 0.25] = 0
    score *= local_tissue

    score_map = normalize_to_01(score)

    return score_map



def visualize_hotspot_overlay(im_rgb, score_map, tissue_mask, percentile = 60, merge_kernel_size=15):

    valid_scores = score_map[tissue_mask > 0]
    if len(valid_scores) == 0 or np.max(valid_scores) <= 1e-8:
        return im_rgb.copy()
    thresh = np.percentile(valid_scores, percentile)
    #roi_mask = (score_map >= thresh).astype(np.uint8)
    roi_mask = ((score_map >= thresh) & (score_map > 0) & (tissue_mask > 0)).astype(np.uint8)
    

    kernel = np.ones((merge_kernel_size, merge_kernel_size), np.uint8)
    # roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)
    # roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel)

    overlay = im_rgb.copy()

    overlay[roi_mask > 0] = [0, 255, 0]   # green

    alpha = 0.35
    result = cv2.addWeighted(overlay, alpha, im_rgb, 1 - alpha, 0)

    return result





def build_roi_overlay(im_rgb, max_side=512):
    if im_rgb is None:
        return None

    img = im_rgb.copy()

    h, w = img.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale < 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    tissue_mask = get_tissue_mask(img)
    _, imH, imE = get_he_deconvolution(img)
    score_map = compute_suspicious_score_map(img, imH, imE, tissue_mask)
    overlay = visualize_hotspot_overlay(img, score_map, tissue_mask)

    return overlay




def get_tissue_mask_global(img, sat_thresh=10, val_thresh=250):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    mask = (sat > sat_thresh) & (val < val_thresh)

    # kernel = np.ones((5, 5), np.uint8)
    # mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    return mask.astype(bool)


##############################################################################
# ROI_Finder Clustering
##############################################################################
def find_clusters(img, model, transform, device,feature_layer = 2, sat_thresh=10, val_thresh=250,
                  patch_size=48, n_clusters=5, batch_size=128, tissue_threshold=0.95):

    h, w, c = img.shape
    h_steps = h // patch_size
    w_steps = w // patch_size

    if h_steps == 0 or w_steps == 0:
        return img

    tissue_mask = get_tissue_mask_global(img, sat_thresh=sat_thresh, val_thresh=val_thresh)
    global_pool = nn.AdaptiveAvgPool2d((1, 1))
    flatten = nn.Flatten()

    patches = []
    valid_indices = []

    for i in range(h_steps):
        for j in range(w_steps):
            patch = img[i*patch_size:(i+1)*patch_size,
                        j*patch_size:(j+1)*patch_size]

            mask_patch = tissue_mask[i*patch_size:(i+1)*patch_size,
                                     j*patch_size:(j+1)*patch_size]

            if np.mean(mask_patch) > tissue_threshold:
                patches.append(patch)
                valid_indices.append((i, j))

    if len(patches) == 0:
        return img

    all_embeddings = []

    with torch.no_grad():
        for start_idx in range(0, len(patches), batch_size):
            batch_patches = patches[start_idx:start_idx + batch_size]

            batch_tensors = [transform(Image.fromarray(p)) for p in batch_patches]
            batch_tensor = torch.stack(batch_tensors).to(device)
            if feature_layer is not None:

                features = model(batch_tensor) 
                feat3 = features[feature_layer]


                x = global_pool(feat3)      
                embedding = flatten(x)      
            else:
                embedding = model(batch_tensor)  

            all_embeddings.append(embedding.cpu().numpy())


    embeddings_matrix = np.concatenate(all_embeddings, axis=0)
    #print("Embedding shape:", embeddings_matrix.shape)

    # ---- Normalize----
    embeddings_matrix = normalize(embeddings_matrix, norm='l2')

    # ---- KMeans ----
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto', **_KMEANS_KW)
    cluster_labels = kmeans.fit_predict(embeddings_matrix) + 1 

    # ---- Rebuild full cluster map ----
    cluster_map = np.full((h_steps, w_steps), 0, dtype=np.int32)

    for idx, (i, j) in enumerate(valid_indices):
        cluster_map[i, j] = cluster_labels[idx]

    cluster_map_norm = (cluster_map * (255 / n_clusters)).astype(np.uint8)

    # ---- Heatmap ----
    heatmap_bgr = cv2.applyColorMap(cluster_map_norm, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    heatmap_resized = cv2.resize(heatmap_rgb, (w, h), interpolation=cv2.INTER_NEAREST)

    blended = cv2.addWeighted(img, 0.6, heatmap_resized, 0.4, 0)

    return blended  