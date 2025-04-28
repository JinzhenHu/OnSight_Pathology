# README

## Folders and Scripts

- **`Retinanet/`** — model architecture **and** post‑processing for mitosis detection.
- **`llm_manager.py`** — loads and chats with five Vision‑Language Models (see table below).
- **`app.spec.py`** - file used for pyinstaller.
## Vision‑Language Models handled by `llm_manager.py`

| Model | Hugging Face link | Note |
|-------|-------------------|------|
| InternVL2‑2B | https://huggingface.co/OpenGVLab/InternVL2-2B | |
| InternVL2‑4B | https://huggingface.co/OpenGVLab/InternVL2-4B | |
| InternVL2‑8B | https://huggingface.co/OpenGVLab/InternVL2-8B | |
| Qwen‑VL‑Chat | https://huggingface.co/Qwen/Qwen-VL-Chat | |
| Bio‑Med Llama‑3‑8B | https://huggingface.co/ContactDoctor/Bio-Medical-Llama-3-8B | *Permission required* |

## Included ML Models

- **RetinaNet** — mitosis detection.
- **VGG‑19 & ViT** — tumor compactness classification.
- **ViT** — glioma classification.
- **YOLO** — MIB segmentation.



