import os
import sys

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


def load_model(model_info):

    res = {
        'model': None,
        'process_region_func': None,
        'using_gpu': False
    }
    ##############################################################################
    #ONNX
    ##############################################################################
    if model_info['repo_src'] == 'HuggingFace':
#############################################################
    # #VIT glio
    # ##############################################################################
        if model_info['model'] == 'VIT':
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(repo_id=model_info['repo'], filename="best_glio.pth")
            import timm
            import torch
            import torch.nn as nn
            from process_region_VIT import process_region
            #Create Model
            model = timm.create_model(
            model_name="hf-hub:1aurent/vit_base_patch16_224.kaiko_ai_towards_large_pathology_fms",
            dynamic_img_size=True,
            pretrained=True,)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            num_features = model.num_features
            num_classes = 3 
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
    #VIT-Tumor-16
    ##############################################################################
        if model_info['model'] == 'VITtumor':
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(repo_id=model_info['repo'], filename="best_kaikuo.pth")
            import timm
            import torch
            import torch.nn as nn
            from process_region_VIT_tumor16 import process_region
            #Create Model
            model = timm.create_model(
            model_name="hf-hub:1aurent/vit_base_patch16_224.kaiko_ai_towards_large_pathology_fms",
            dynamic_img_size=True,
            pretrained=True,)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            num_features = model.num_features
            num_classes = 16
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
    #VIT-Tumor-4-kaiko
    ##############################################################################
        if model_info['model'] == 'VITtumor_kaiko':
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(repo_id=model_info['repo'], filename="best_kaikuo_20000image.pth")
            import timm
            import torch
            import torch.nn as nn
            from process_region_VIT_tumor4_kaiko import process_region
            #Create Model
            model = timm.create_model(
            model_name="hf-hub:1aurent/vit_base_patch16_224.kaiko_ai_towards_large_pathology_fms",
            dynamic_img_size=True,
            pretrained=True,)

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
    #VIT-Tumor-4-optimus
    ##############################################################################
        if model_info['model'] == 'VITtumor_Optimus':
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(repo_id=model_info['repo'], filename="best_optimus_image6000.pth")
            import timm
            import torch
            import torch.nn as nn
            from process_region_VIT_tumor4_optimus  import process_region
            #Create Model
            model = timm.create_model(
                "hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False
            )
            model.eval()

            # Add a classification head
            num_features = model.num_features
            num_classes = 4
            model.head = nn.Sequential(
                nn.Linear(num_features, num_classes),  
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            res['model'] = model
            res['process_region_func'] = process_region
            res['using_gpu'] = torch.cuda.is_available()
    ##############################################################################
    #Resnet-Tumor-4
    ##############################################################################
        if model_info['model'] == 'Resnet_Tumor':
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(repo_id=model_info['repo'], filename="resnet20000.pth")
            import timm
            import torch
            import torch.nn as nn
            from process_region_Resnet_tumor4 import process_region
            #Create Model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = timm.create_model(
                'resnet101.a1h_in1k',
                pretrained=True,
                features_only=False,
            )
            # model = model.eval()

            # get model specific transforms (normalization, resize)
            # data_config = timm.data.resolve_model_data_config(model)
            # transforms = timm.data.create_transform(**data_config, is_training=True)
            # num_features = model.num_features
            # num_classes = 4 ### Three classes
            model.fc = nn.Sequential(
                nn.Linear(2048, 256),
                nn.ReLU(),
                # nn.Dropout(0.3),
                nn.Linear(256, 4),
            )
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            res['model'] = model
            res['process_region_func'] = process_region
            res['using_gpu'] = torch.cuda.is_available()
    ##############################################################################
    #Retinanet
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
    #YOLO
    ##############################################################################
        if model_info['model'] == 'YOLO':
            from ultralytics import YOLO
            import torch
            from process_region_YOLO import process_region
            from huggingface_hub import hf_hub_download

            #weights_path = hf_hub_download(repo_id=model_info['repo'], filename="best (11x seg new).pt")
            weights_path = hf_hub_download(repo_id=model_info['repo'], filename="best.pt")


            res['model'] = YOLO(weights_path)
            res['process_region_func'] = process_region
            res['using_gpu'] = torch.cuda.is_available()

            return res
    
        if model_info['model'] == 'YOLO_det':
            from ultralytics import YOLO
            import torch
            from process_region_YOLO_det import process_region
            from huggingface_hub import hf_hub_download

            weights_path = hf_hub_download(repo_id=model_info['repo'], filename="best (12x det new).pt")


            res['model'] = YOLO(weights_path)
            res['process_region_func'] = process_region
            res['using_gpu'] = torch.cuda.is_available()

            return res

    elif model_info['repo_src'] == 'Local':
        pass
    

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

