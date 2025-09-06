import json

from utils import resource_path

##############################################################################################################################################
# VLM Model List
##############################################################################################################################################

LLM_CATALOG = {
    # "Internvl3-2b(new)":      "metadata/llm_internvl3_2b.json",
    # "Internvl3-8b(new)":      "metadata/llm_internvl3_8b.json",
    "Huatuo-7b": "metadata/huatuo.json",
    "Lingshu-7b": "metadata/lingshu.json",
    # "Internvl2-2b(old)":      "metadata/llm_internvl2_2b.json",
    # "Internvl2-8b(old)":      "metadata/llm_internvl2_8b.json",
    # "Qwen-VL-Chat":      "metadata/llm_qwen.json",
    # "Bio-Medical LLaMA-3":  "metadata/llm_biomed_llama3.json",
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
        # {'name': "Tumor Compact (VGG19)", 'info_file': 'metadata/tumor_compact_vgg.json'},
        # {'name': "Tumor Compact (EfficientNetV2) (Test)", 'info_file': 'metadata/tumor_compact_efficientnet.json'},
        # {'name': "Prior 16-class Tumor Compact (VIT)", 'info_file': 'metadata/tumor_compact_vit.json'},
        {'name': "Tumor 4-Class (VIT)", 'info_file': 'metadata/tumor_compact_kaiko_vit.json'},
        # {'name': "New Tumor 4-Class (Resnet)", 'info_file': 'metadata/tumor_compact_resnet.json'},
        # {'name': "GliomaAstroOligo(VIT)", 'info_file': 'metadata/glio_vit.json'}
    ]),
    ("▶️ Segmentation Models", [
        {'name': "MIB (YOLO)", 'info_file': 'metadata/mib_yolo_1024.json'},
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
