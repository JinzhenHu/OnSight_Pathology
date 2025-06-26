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
        if model_info['model'] == 'ONNX':

            from huggingface_hub import hf_hub_download
            import torch
            import onnxruntime as ort

            model_path = hf_hub_download(repo_id=model_info['repo'], filename="model.onnx")

            # Load ONNX model with GPU support if available
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                providers = ['CUDAExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

            session = ort.InferenceSession(model_path, providers=providers)

            from process_region_onnx import process_region

            res['model'] = session
            res['process_region_func'] = process_region
            res['using_gpu'] = 'CUDAExecutionProvider' in available_providers

            return res
    # ##############################################################################
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
            Cl = _safe_float(model_info['additional_configs'].get('confidence_level', 0))

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

    ##############################################################################
    # Mask-RCNN Detectron2
    ##############################################################################

        if model_info['model'] == 'MaskRCNN':
            import torch
            from detectron2.engine import DefaultPredictor
            from detectron2.config import get_cfg
            from detectron2 import model_zoo
            from huggingface_hub import hf_hub_download
            from process_region_MRCNN_MIB import process_region

            weights_path = hf_hub_download(repo_id=model_info['repo'], filename="model.pth")

            cfg = get_cfg()
            cfg.merge_from_file(resource_path("detectron2/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
            cfg.MODEL.WEIGHTS = weights_path
            cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

            model = DefaultPredictor(cfg)

            res['model'] = model
            res['process_region_func'] = process_region
            res['using_gpu'] = torch.cuda.is_available()

            return res

    elif model_info['repo_src'] == 'Local':
        pass
    

    return res