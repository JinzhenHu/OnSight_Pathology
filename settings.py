import json

from utils import resource_path

##############################################################################################################################################
# VLM Model List
##############################################################################################################################################

LLM_CATALOG = {
    "Huatuo-7b": "metadata/huatuo.json",
    "Lingshu-7b": "metadata/lingshu.json",
}

LLM_METADATA = {}
for name, path in LLM_CATALOG.items():
    with open(resource_path(path), "r", encoding="utf-8") as f:
        LLM_METADATA[name] = json.load(f)

PRECISION_DISPLAY_MAP = {
    "4bit": "Fastest Speed",
    "8bit": "Balanced",
    "16bit": "Highest Quality"
}
##############################################################################################################################################
# Dropdown model metadata
##############################################################################################################################################

MODEL_CATELOG = [
    ("▶️ Classification Models", [
        {'name': "Tumor 4-Class (VIT)", 'info_file': 'metadata/tumor_compact_kaiko_vit.json'},
    ]),
    ("▶️ Segmentation Models", [
        {'name': "MIB (YOLO)", 'info_file': 'metadata/mib_yolo_1024.json'},
        {'name': "MIB (Cellpose-SAM)", 'info_file': 'metadata/mib_cellpose.json'},
    ]),
    ("▶️ Object Detection Models", [
        {'name': "Mitosis Detection (Retinanet)", 'info_file': 'metadata/mib_mitosis.json'}
    ]),
]

MODEL_METADATA = {}
for _, _v in MODEL_CATELOG:
    for __v in _v:
        with open(resource_path(__v['info_file'])) as f:
            MODEL_METADATA[__v['name']] = json.load(f)
