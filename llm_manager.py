# llm_manager.py
import os
import json, torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig,AutoModelForCausalLM, TextIteratorStreamer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from lemon import model
from utils import resource_path
import tempfile
from threading import Thread
from contextlib import contextmanager
from qwen_vl_utils import process_vision_info 
#########################################################################################################
#Loading LLM 
#########################################################################################################
#@lru_cache(maxsize=None) This allows the model stays in cache but will makes the software stuck if the user didn't have a large ARM GPU
def load_llm(config_path: str, precision: str = "8bit"):
    """
    Return (model, tokenizer, cfg) — cached by config_path.
    """
    with open(resource_path(config_path), "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
        
    # ---------- HuatuoGPT-Vision & Lingshu-7B ----------------
    repo_lower = cfg["repo"].lower()
    # ---------- HuatuoGPT‑Vision‑7B‑Qwen2.5VL & Lingshu-7B ---------------------------
    if repo_lower.endswith("huatuogpt-vision-7b-qwen2.5vl") or repo_lower.endswith("lingshu-7b"):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
        }

        if precision == "4bit":
            # For 4-bit
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif precision == "16bit":
            # For 16-bit
            model_kwargs["torch_dtype"] = torch.bfloat16
        else: # Default to 8bit
            # For 8-bit
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg["repo"],
            **model_kwargs,
        ).eval()
        
        tokenizer = AutoProcessor.from_pretrained(cfg["repo"])
        return model, tokenizer, cfg


#########################################################################################################
# Chat with LLM & Advanced Memory Management
#########################################################################################################
def _compress_memory(model, tokenizer, history):
    """
    [内部函数] 后台无声压缩记忆：保留首图和最新问题，总结中间对话
    """
    turn_1 = history[:2]
    current_turn = history[-1:]
    middle_history = history[2:-1]

    summary_prompt = "Please summarize the following conversation concisely, keeping key medical facts:\n"
    for msg in middle_history:
        role = msg["role"]
        # 安全提取文本内容
        text_content = next((item["text"] for item in msg["content"] if item.get("type") == "text"), "")
        summary_prompt += f"{role.upper()}: {text_content}\n"

    summary_messages = [{"role": "user", "content": [{"type": "text", "text": summary_prompt}]}]
    
    text = tokenizer.apply_chat_template(summary_messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text=[text], padding=True, return_tensors="pt").to(model.device)

    # 瞬间生成总结 (不推流)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256, eos_token_id=tokenizer.tokenizer.eos_token_id)
        
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    summary = tokenizer.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    summary_msg = {
        "role": "system",
        "content": [{"type": "text", "text": f"[Past Conversation Summary]: {summary}"}]
    }
    
    return turn_1 + [summary_msg] + current_turn


#########################################################################################################
#Chat with LLM
#########################################################################################################
def stream_reply(model, tokenizer, llm_config, pil_img, prompt, history, streamer_callback=None, has_new_image=False):
    """
    Returns (assistant_reply_text, updated_history)
    """
    repo_lower = llm_config["repo"].lower()
    
    if repo_lower.endswith("huatuogpt-vision-7b-qwen2.5vl") or repo_lower.endswith("lingshu-7b"):
        MAX_TURNS = 5 

        # ==========================================
        # 1. 构建增量 History (Visual Anchoring)
        # ==========================================
        # 🚀 [修改逻辑]：如果是第一轮，或者是用户刚点了 Update 按钮换了新图
        if len(history) == 0 or has_new_image:
            history.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img}, # 塞入最新图片
                    {"type": "text",  "text": prompt},
                ],
            })
        else:
            # 否则，依旧采用省显存的纯文字交互
            history.append({
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            })
        # ==========================================
        # 2. 触发滚动记忆压缩
        # ==========================================
        user_msg_count = sum(1 for msg in history if msg["role"] == "user")
        if user_msg_count > MAX_TURNS:
            if streamer_callback:
                streamer_callback("\n<i style='color:gray;'>[System: Compressing old memory to save VRAM...]</i><br>\n")
            history = _compress_memory(model, tokenizer, history)

        # ==========================================
        # 3. 准备模型推理
        # ==========================================
        text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(history)

        inputs = tokenizer(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        streamer = TextIteratorStreamer(tokenizer.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=1024,
            eos_token_id=tokenizer.tokenizer.eos_token_id
        )
        
        txt = ""
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # 实时推流给前端
        for new_text in streamer:
            # 解决一些模型输出的换行符在 HTML 里不显示的问题
            formatted_text = new_text.replace("\n", "<br>")
            if streamer_callback:
                streamer_callback(formatted_text)
            txt += formatted_text
        
        thread.join() 
        
        # ==========================================
        # 4. 保存助手回复并返回
        # ==========================================
        history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": txt}]
        })
        
        return txt, history