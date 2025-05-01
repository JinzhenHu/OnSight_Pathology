# llm_manager.py
import os
from functools import lru_cache
import json, torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig,AutoModelForCausalLM
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from utils import resource_path
from timm.models.layers import DropPath
from pathlib import Path
from transformers.generation import GenerationConfig
import tempfile
from contextlib import contextmanager
#########################################################################################################
#Loading LLM 
#########################################################################################################
#@lru_cache(maxsize=None) This allows the model stays in cache but will makes the software stuck if the user didn't have a large ARM GPU
def load_llm(config_path: str):
    """
    Return (model, tokenizer, cfg) — cached by config_path.
    """
    with open(resource_path(config_path), "r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    # ---------- QwenVL -8B ---------------------------
    if cfg["repo"].lower().endswith("qwen-vl-chat"):
        tokenizer = AutoTokenizer.from_pretrained(cfg["repo"], trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(cfg["repo"], device_map="cuda", trust_remote_code=True).eval()
        return model, tokenizer, cfg 


    # ---------- Intern VL 2-8B ---------------------------
    if cfg["repo"].lower().endswith("internvl2-8b"):
        model = AutoModel.from_pretrained(
            cfg["repo"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            cfg.get("tokenizer_repo", cfg["repo"]),
            trust_remote_code=True,
            use_fast=False,
        )
        return model, tokenizer, cfg  
    

    # ---------- Intern VL 2-4B ---------------------------
    if cfg["repo"].lower().endswith("internvl2-4b"):
        model = AutoModel.from_pretrained(
            cfg["repo"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            cfg.get("tokenizer_repo", cfg["repo"]),
            trust_remote_code=True,
            use_fast=False,
        )
        return model, tokenizer, cfg   
    

    # ---------- Intern VL 2-2B ---------------------------
    if cfg["repo"].lower().endswith("internvl2-2b"):
        model = AutoModel.from_pretrained(
            cfg["repo"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            cfg.get("tokenizer_repo", cfg["repo"]),
            trust_remote_code=True,
            use_fast=False,
        )
        return model, tokenizer, cfg   # 
    
    # ----------  Bio-Med Llama-3 -------------
    if cfg["repo"].lower().endswith("bio-medical-multimodal-llama-3-8b-v1"):
        qcfg = BitsAndBytesConfig(**cfg["bitsandbytes"]) if cfg.get("bitsandbytes") else None
        model = AutoModel.from_pretrained(
            cfg["repo"],
            torch_dtype=getattr(torch, cfg.get("dtype", "float16")),
            device_map="auto",
            quantization_config=qcfg,
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            cfg.get("tokenizer_repo", cfg["repo"]),
            trust_remote_code=True,
        )

    return model, tokenizer, cfg

#########################################################################################################
#Chat with LLM
#########################################################################################################
def stream_reply( model, tokenizer,llm_config, pil_img, prompt, history):
    """
    Returns (assistant_reply_text, updated_history)

    Handles:
      • LLaMA/LLaVA style 
      • InternVL style  
      • Qwen style
    """



#########################################################################################################
    #Below is the helper Function for InternVL
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def build_transform(input_size):
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(image, input_size=448, max_num=12):
        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
 #########################################################################################################
    # ----------  QwenVL Chat -------------
    if llm_config["repo"].lower().endswith("qwen-vl-chat"):

        @contextmanager
        def pil_to_qwen_path(pil_img, fmt="PNG"):
            fd, tmp_path = tempfile.mkstemp(suffix="." + fmt.lower())
            os.close(fd)
            pil_img.save(tmp_path, format=fmt)
            try:
                yield tmp_path        
            finally:
                try:
                    os.remove(tmp_path)
                except FileNotFoundError:
                    pass
        
        with pil_to_qwen_path(pil_img) as path:
            query = tokenizer.from_list_format([
                {"image": path},
                {"text": prompt}
            ])
            reponse, history = model.chat(tokenizer, query=query, history=history)
  

        txt =reponse
        history.append({"role": "user", "content": [pil_img, prompt]})
        history.append({"role": "assistant", "content": reponse})


    # ----------  Bio-Med Llama-3 -------------
    if llm_config["repo"].lower().endswith("bio-medical-multimodal-llama-3-8b-v1"):
        history.append({"role": "user", "content": [pil_img, prompt]})
        txt = ""
        for chunk in model.chat(
                image=pil_img,
                msgs=history,
                tokenizer=tokenizer,
                sampling=True,
                temperature=0.95,
                stream=True):
            txt += chunk
        history.append({"role": "assistant", "content": txt})

        
    # ----------  InternVL2-8b -------------
    if llm_config["repo"].lower().endswith("internvl2-8b"):
        generation_cfg = dict(max_new_tokens=1024, do_sample=True)
        pixel_values = load_image(pil_img, max_num=12).to(torch.bfloat16).cuda()
        question = "<image>\n" + prompt       

        raw_reply = model.chat(tokenizer, pixel_values, question, generation_cfg)

        clean_txt = f"\n {raw_reply}"  
        txt = clean_txt.replace("\n", "<br>")                 


        history.append({"role": "user", "content": [pil_img, prompt]})
        history.append({"role": "assistant", "content": raw_reply})

    # ----------  InternVL2-4b -------------
    if llm_config["repo"].lower().endswith("internvl2-4b"):

        generation_cfg = dict(max_new_tokens=1024, do_sample=True)
        pixel_values = load_image(pil_img, max_num=12).to(torch.bfloat16).cuda()
        question = "<image>\n" + prompt       

        raw_reply = model.chat(tokenizer, pixel_values, question, generation_cfg)

        clean_txt = f"\n {raw_reply}"  
        txt = clean_txt.replace("\n", "<br>")                 


        history.append({"role": "user", "content": [pil_img, prompt]})
        history.append({"role": "assistant", "content": raw_reply})

    # ----------  InternVL2-4b -------------
    if llm_config["repo"].lower().endswith("internvl2-2b"):

        generation_cfg = dict(max_new_tokens=1024, do_sample=True)
        pixel_values = load_image(pil_img, max_num=12).to(torch.bfloat16).cuda()
        question = "<image>\n" + prompt       


        raw_reply = model.chat(tokenizer, pixel_values, question, generation_cfg)

        clean_txt = f"\n {raw_reply}"  
        txt = clean_txt.replace("\n", "<br>")                 


        history.append({"role": "user", "content": [pil_img, prompt]})
        history.append({"role": "assistant", "content": raw_reply})

    return txt , history






