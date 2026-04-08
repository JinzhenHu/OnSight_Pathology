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
from NuLite.models.nulite import NuLite
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# Always use this for accessing any local path
def resource_path(relative_path):
    """Get absolute path to resource (for dev and for PyInstaller onefile mode)"""
    if hasattr(sys, '_MEIPASS'):
        # _MEIPASS is the temp folder where PyInstaller unpacks files
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


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

def load_nulite_weights(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "config" in ckpt:
        run_conf = ckpt["config"]

        backbone = run_conf["model.backbone"]
        num_nuclei_classes = run_conf["data.num_nuclei_classes"]
        num_tissue_classes = run_conf["data.num_tissue_classes"]
        mean = run_conf.get("transformations.normalize.mean", [0.5, 0.5, 0.5])
        std = run_conf.get("transformations.normalize.std", [0.5, 0.5, 0.5])
        nuclei_types = run_conf.get(
            "dataset_config.nuclei_types",
            {
                "Background": 0, "Neoplastic": 1, "Inflammatory": 2,
                "Connective": 3, "Dead": 4, "Epithelial": 5,
            },
        )
        tissue_types = run_conf.get("dataset_config.tissue_types", None)

    # Need to make sure NuLite is imported here if not at the top of the file
    from NuLite.models.nulite import NuLite
    
    model = NuLite(
        num_nuclei_classes=num_nuclei_classes,
        num_tissue_classes=num_tissue_classes,
        vit_structure=backbone,
    )

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    model.to(device)

    type_id_to_name = {v: k for k, v in nuclei_types.items()}
    tissue_id_to_name = {v: k for k, v in tissue_types.items()} if tissue_types else None

    return model, mean, std, type_id_to_name, tissue_id_to_name, device

class MidnightClassifier(nn.Module):
    def __init__(self, model_name="kaiko-ai/midnight", num_classes=2, dropout=0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size  # 1536

        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, num_classes),
            # nn.GELU(),
            # nn.Dropout(dropout),
            # nn.Linear(512, num_classes)
        )

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        tokens = outputs.last_hidden_state          # [B, 1+N, 1536]
        cls_token = tokens[:, 0, :]                # [B, 1536]
        patch_mean = tokens[:, 1:, :].mean(dim=1) # [B, 1536]
        feat = torch.cat([cls_token, patch_mean], dim=1)  # [B, 3072]
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
            model_path = hf_hub_download(repo_id=model_info['repo'], filename="best_kaikuo_20000image.pth")
            import timm
            import torch
            import torch.nn as nn
            from process_region_VIT_tumor4_kaiko import process_region
            # Create Model
            model = timm.create_model(
                model_name="hf-hub:1aurent/vit_base_patch16_224.kaiko_ai_towards_large_pathology_fms",
                dynamic_img_size=True,
                pretrained=True, )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            res['using_gpu'] = torch.cuda.is_available()
        ##############################################################################
        # VIT-Magnification-4-kaiko
        ##############################################################################
        if model_info['model'] == 'VITmagnification_kaiko':
            from huggingface_hub import hf_hub_download
            #model_path = r"D:\UofT\2025fall\OnSight\Revisions\Mag_Detector\best_model.pth"
            model_path = hf_hub_download("diamandislabii/Magnification_Detection", filename="best_model.pth")
            import timm
            import torch
            import torch.nn as nn
            from process_region_mag_detector import process_region
            # Create Model
            model = timm.create_model(
            model_name="hf-hub:1aurent/vit_small_patch16_224.kaiko_ai_towards_large_pathology_fms",
               # dynamic_img_size=True,
                pretrained=True, )


            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            res['using_gpu'] = torch.cuda.is_available()
        ##############################################################################
        # VIT-Glio-Subtype-kaiko
        ##############################################################################
        if model_info['model'] == 'Glioma_Subtype':
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(repo_id=model_info['repo'], filename="best_model_binary.pth")
            
            import timm
            import torch
            import torch.nn as nn
            from process_region_VIT_glioma import process_region
            # Create Model
            model = timm.create_model(
                model_name="hf-hub:1aurent/vit_base_patch16_224.kaiko_ai_towards_large_pathology_fms",
                #dynamic_img_size=True,
                pretrained=True, )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            num_features = model.num_features
            num_classes = 2
            # model.head = nn.Sequential(
            #     nn.LayerNorm(num_features),  
            #     nn.Linear(num_features, 256),  
            #     nn.GELU(),                  
            #     nn.Dropout(0.5),              
            #     nn.Linear(256, num_classes)    
            # )
            # model.head = nn.Sequential(
            #     nn.Linear(num_features, num_classes),
            # )
            #model.head = nn.Linear(num_features, 5)
            model.head = nn.Sequential(
                    nn.Linear(num_features, 512),
                    nn.BatchNorm1d(512),
                    nn.GELU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, num_classes)
                )
           #model_path = r"D:\UofT\2025fall\AutomateCell\Spatial\Spatial_Analysis\CellLabeling\best_model.pth"
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            res['model'] = model
            res['process_region_func'] = process_region
            res['using_gpu'] = torch.cuda.is_available()
        ##############################################################################
        # VIT-Glio-Prov-Gigapath-kaiko
        ##############################################################################
        if model_info['model'] == 'Glioma_Subtype_Astro_G4':
            from huggingface_hub import hf_hub_download
            #model_path = r"D:\UofT\2025fall\OnSight\Revisions\Glioma_Subtype\best_model_mutant_binary.pth"
            #model_path = r"D:\UofT\2025fall\OnSight\Revisions\Glioma_Subtype\best_model_mutant_binary_kaiko.pth"
            model_path = hf_hub_download(repo_id=model_info['repo'], filename="best_model_mutant_binary_kaiko_60000.pth")

            
            import timm
            import torch
            import torch.nn as nn
            from process_region_VIT_glioma import process_region
            # Create Model
            #model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
            model = timm.create_model(
            model_name="hf-hub:1aurent/vit_base_patch16_224.kaiko_ai_towards_large_pathology_fms",
            #dynamic_img_size=True,
            pretrained=True,
            ).eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            num_features = model.num_features
            num_classes = 2
            # model.head = nn.Sequential(
            #     nn.LayerNorm(num_features),  
            #     nn.Linear(num_features, 256),  
            #     nn.GELU(),                  
            #     nn.Dropout(0.5),              
            #     nn.Linear(256, num_classes)    
            # )
            # model.head = nn.Sequential(
            #     nn.Linear(num_features, num_classes),
            # )
            #model.head = nn.Linear(num_features, 5)
            model.head = nn.Sequential(
                nn.Linear(num_features, num_classes),
                # nn.GELU(),
                # nn.Dropout(0.3),
                # nn.Linear(512, num_classes)
            )
            #model_path = r"D:\UofT\2025fall\AutomateCell\Spatial\Spatial_Analysis\CellLabeling\best_model.pth"
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            res['model'] = model
            res['process_region_func'] = process_region
            res['using_gpu'] = torch.cuda.is_available()        

       ##############################################################################
        # VIT-Glio-Prov-Gigapath-kaiko-threeclass
        ##############################################################################
        if model_info['model'] == 'Midnight_Glioma':
            from huggingface_hub import hf_hub_download
            model_path = r"D:\UofT\2025fall\OnSight\Revisions\Glioma_Subtype\best_model_mutant_binary_kaiko.pth"
            #model_path = hf_hub_download(repo_id=model_info['repo'], filename="best_model.pth")

            
            import timm
            import torch
            import torch.nn as nn
            from process_region_VIT_midnight import process_region
            # Create Model
            #model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
            # model = timm.create_model(
            # model_name="hf-hub:1aurent/vit_base_patch16_224.kaiko_ai_towards_large_pathology_fms",
            # #dynamic_img_size=True,
            # pretrained=True,
            # ).eval()
            num_classes = 2
            #model = MidnightClassifier(num_classes=num_classes)
            model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            num_features = model.num_features

            model.head = nn.Sequential(
                nn.Linear(num_features, num_classes),
                # nn.GELU(),
                # nn.Dropout(0.3),
                # nn.Linear(512, num_classes)
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            #num_features = model.num_features
            
            # model.head = nn.Sequential(
            #     nn.LayerNorm(num_features),  
            #     nn.Linear(num_features, 256),  
            #     nn.GELU(),                  
            #     nn.Dropout(0.5),              
            #     nn.Linear(256, num_classes)    
            # )
            # model.head = nn.Sequential(
            #     nn.Linear(num_features, num_classes),
            # )
            #model.head = nn.Linear(num_features, 5)
            # model.head = nn.Sequential(
            #     nn.Linear(num_features, 512),
            #     nn.GELU(),
            #     nn.Dropout(0.3),
            #     nn.Linear(512, num_classes)
            # )
            #model_path = r"D:\UofT\2025fall\AutomateCell\Spatial\Spatial_Analysis\CellLabeling\best_model.pth"
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            res['model'] = model
            res['process_region_func'] = process_region
            res['using_gpu'] = torch.cuda.is_available()        

        ##############################################################################
        # CellViT
        ##############################################################################
        if model_info['model'] == 'CellViT':
            import torch
            from NuLite.models.nulite import NuLite
            from huggingface_hub import hf_hub_download
            from process_region_cellvit import process_region

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_path = r"D:\Downloads\NuLite-H-Weights.pth"
            model_file, mean, std, type_id_to_name, tissue_id_to_name, device = load_nulite_weights(model_path, device)
            model = [model_file, mean, std, type_id_to_name, tissue_id_to_name, device]



            res['model'] = model
            res['process_region_func'] = process_region
            res['using_gpu'] = torch.cuda.is_available()     





        ##############################################################################
        # Retinanet
        ##############################################################################
        if model_info['model'] == 'Retinanet':
            import torch
            import yaml
            from retinanet.Model import MyMitosisDetection
            from huggingface_hub import hf_hub_download

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model_path = hf_hub_download(repo_id=model_info['repo'], filename="bestmodel.pth")
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
            res['using_gpu'] = torch.cuda.is_available()

            return res

        ##############################################################################
        # Cellpose-SAM
        ##############################################################################
        if model_info['model'] == 'Cellpose-SAM':
            from cellpose import models
            import torch
            from process_region_cellpose import process_region

            res['model'] = models.CellposeModel(gpu=True) if torch.cuda.is_available() else models.CellposeModel(
                gpu=False)
            res['process_region_func'] = process_region
            res['using_gpu'] = torch.cuda.is_available()

        ##############################################################################
        # YOLO
        ##############################################################################
        if model_info['model'] == 'YOLO':
            from ultralytics import YOLO
            import torch
            from process_region_YOLO import process_region
            from huggingface_hub import hf_hub_download

            weights_path = hf_hub_download(repo_id=model_info['repo'], filename="best.pt")

            res['model'] = YOLO(weights_path)
            res['process_region_func'] = process_region
            res['using_gpu'] = torch.cuda.is_available()

            return res
        ##############################################################################
        # InstanSeg
        ##############################################################################
    if model_info['model'] == 'InstanSeg':
        from instanseg import InstanSeg
        import torch
        from process_region_instanseg import process_region

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # InstanSeg 自动管理下载，不需要 HuggingFace Hub Download
        model = InstanSeg(model_type="brightfield_nuclei", device=device)
        
        res['model'] = model
        res['process_region_func'] = process_region
        res['using_gpu'] = torch.cuda.is_available()
        return res
    if model_info['model'] == 'CPSAM_profiler':
        from cellpose import models
        import torch
        from process_region_cpsam_cellfeatures import process_region

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # InstanSeg 自动管理下载，不需要 HuggingFace Hub Download
        model = models.CellposeModel(gpu=True)
        
        res['model'] = model
        res['process_region_func'] = process_region
        res['using_gpu'] = torch.cuda.is_available()
        return res
    if model_info['model'] == 'CPSAM':
        from cellpose import models
        import torch
        from process_region_cpsam import process_region

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # InstanSeg 自动管理下载，不需要 HuggingFace Hub Download
        model = models.CellposeModel(gpu=True)
        
        res['model'] = model
        res['process_region_func'] = process_region
        res['using_gpu'] = torch.cuda.is_available()
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

# roi_overlay_utils.py


def normalize_to_01(arr):
    arr = arr.astype(np.float32)
    mn, mx = np.min(arr), np.max(arr)
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)

def get_tissue_mask(im_rgb, min_tissue_area=500, kernel_size=5):
    """
    从低倍 thumbnail 中提取组织区域，去掉空白背景。
    """
    gray = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2GRAY)

    # Otsu 反阈值：背景亮，组织暗
    _, tissue_mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    tissue_mask = tissue_mask.astype(np.uint8)

    # 形态学清理
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)

    # 去掉很小的碎片
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(tissue_mask, 8)
    cleaned = np.zeros_like(tissue_mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_tissue_area:
            cleaned[labels == i] = 255

    return (cleaned > 0).astype(np.uint8)


def get_he_deconvolution(im_rgb):
    """
    使用 HistomicsTK 标准 H&E stain matrix 做颜色反卷积。
    """
    stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
    W = np.array([
        stain_color_map['hematoxylin'],
        stain_color_map['eosin'],
        stain_color_map['null']
    ]).T

    deconv = htk.preprocessing.color_deconvolution.color_deconvolution(im_rgb, W)

    # HistomicsTK 的 Stains 通常是便于显示的 0-255 图
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

    # H/E 强度：越大表示染色越重
    H_strength = (255.0 - imH) * tissue_float
    E_strength = (255.0 - imE) * tissue_float

    # RGB 通道
    R = im_rgb[:, :, 0].astype(np.float32) * tissue_float
    G = im_rgb[:, :, 1].astype(np.float32) * tissue_float
    B = im_rgb[:, :, 2].astype(np.float32) * tissue_float


    # 归一化
    Hn = normalize_to_01(H_strength)
    En = normalize_to_01(E_strength)
    #Tn = normalize_to_01(texture_map)

    # # 区域级平滑
    H_region = cv2.GaussianBlur(Hn, (0, 0), sigmaX=sigma_region, sigmaY=sigma_region)
    E_region = cv2.GaussianBlur(En, (0, 0), sigmaX=sigma_region, sigmaY=sigma_region)
    # T_region = cv2.GaussianBlur(Tn, (0, 0), sigmaX=sigma_region, sigmaY=sigma_region)

    # 红色惩罚：E 高但 H 不高，容易是假阳性
    red_penalty = np.clip(E_region - H_region, 0, 1)


    local_tissue = cv2.GaussianBlur(tissue_float, (0, 0), sigmaX=sigma_region, sigmaY=sigma_region)

    # 新分数：轻 惩罚红区、奖励蓝紫偏向
    score = (
       1 * H_region -
       # 0.15 * T_region +
       # 0.20 * rb_region -
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
    """
    输入 RGB 图，输出 RGB overlay 图。
    为了 overview 更流畅，这里先缩放后算。
    """
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

    # ---- Extract patches (only tissue) ----
    for i in range(h_steps):
        for j in range(w_steps):
            patch = img[i*patch_size:(i+1)*patch_size,
                        j*patch_size:(j+1)*patch_size]

            mask_patch = tissue_mask[i*patch_size:(i+1)*patch_size,
                                     j*patch_size:(j+1)*patch_size]

            # 只保留有组织的 patch
            if np.mean(mask_patch) > tissue_threshold:
                patches.append(patch)
                valid_indices.append((i, j))

    if len(patches) == 0:
        return img

    # ---- Extract features ----
    all_embeddings = []

    with torch.no_grad():
        for start_idx in range(0, len(patches), batch_size):
            batch_patches = patches[start_idx:start_idx + batch_size]

            batch_tensors = [transform(Image.fromarray(p)) for p in batch_patches]
            batch_tensor = torch.stack(batch_tensors).to(device)
            if feature_layer is not None:

                features = model(batch_tensor)  # (B, D)
                feat3 = features[feature_layer]
                #print(f"Batch {start_idx//batch_size}: feature shape {feat3.shape}")

                x = global_pool(feat3)      # 形状变成 [B, 960, 1, 1]
                embedding = flatten(x)      # 形状变成 [B, 960]
            else:
                embedding = model(batch_tensor)  # (B, D)

            all_embeddings.append(embedding.cpu().numpy())


    embeddings_matrix = np.concatenate(all_embeddings, axis=0)
    print("Embedding shape:", embeddings_matrix.shape)

    # ---- Normalize（关键！）----
    embeddings_matrix = normalize(embeddings_matrix, norm='l2')

    # ---- KMeans ----
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    # 给聚类标签 +1，让它们变成 1, 2, 3, 4, 5，把 0 留给背景
    cluster_labels = kmeans.fit_predict(embeddings_matrix) + 1 

    # ---- Rebuild full cluster map ----
    cluster_map = np.full((h_steps, w_steps), 0, dtype=np.int32) # 背景直接初始化为 0

    for idx, (i, j) in enumerate(valid_indices):
        cluster_map[i, j] = cluster_labels[idx]

    # ---- Normalize to 0-255 ----
    # 最大值现在是 n_clusters，所以除以 n_clusters
    cluster_map_norm = (cluster_map * (255 / n_clusters)).astype(np.uint8)

    # ---- Heatmap ----
    heatmap_bgr = cv2.applyColorMap(cluster_map_norm, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    heatmap_resized = cv2.resize(heatmap_rgb, (w, h), interpolation=cv2.INTER_NEAREST)

    # ---- Overlay (优化版：只在有组织的区域叠加) ----
    # 将二维的 tissue_mask 扩展为三维，以便与图片通道匹配
    #mask_3d = np.expand_dims(tissue_mask, axis=-1)
    
    # 叠加效果
    blended = cv2.addWeighted(img, 0.6, heatmap_resized, 0.4, 0)
    
    # 使用 np.where，只有在 mask 为 True 的地方用叠加后的图，背景保留原图
   # overlay_img = np.where(mask_3d, blended, img)

    return blended  