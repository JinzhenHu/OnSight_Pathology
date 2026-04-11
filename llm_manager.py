# llm_manager.py
import os
import sys
import json
import torch
from threading import Thread
from typing import List, Dict, Any, Optional, Tuple

from transformers import BitsAndBytesConfig, TextIteratorStreamer, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from utils import resource_path

class AgentMemoryBot:
    """
    高级单图记忆系统：
    1. 首图永久保留
    2. 近期滑动窗口无损保留
    3. 远期记忆调用 LLM 生成智能摘要
    4. 具备换图状态感知与防复读机功能
    """
    def __init__(self, config_path: str, precision: str = "16bit"):
        with open(resource_path(config_path), "r", encoding="utf-8") as fh:
            self.cfg = json.load(fh)

        model_id = self.cfg["repo"]
        # 🚀 修复：将普通日志输出到 stderr，防止污染 stdout 的 JSON 通道
        print(f"[System] Loading model: {model_id}", file=sys.stderr)

        self.processor = AutoProcessor.from_pretrained(model_id)

        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
        }

        if precision == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif precision == "16bit":
            model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, **model_kwargs
        ).eval()

        # =========================
        # 记忆体状态与参数读取
        # =========================
        self.history: List[Dict[str, Any]] = []
        self.memory_summary: str = ""
        self.first_image_message: Optional[Dict[str, Any]] = None

        self.max_input_tokens = self.cfg.get("max_input_tokens", 1500)
        self.max_new_tokens = self.cfg.get("max_new_tokens", 256)
        self.summary_max_new_tokens = self.cfg.get("summary_max_new_tokens", 128)
        self.recent_message_budget = self.cfg.get("recent_message_budget", 4)
        self.summary_char_limit = self.cfg.get("summary_char_limit", 400)
        
        self.system_prompt = self.cfg.get(
            "system_prompt", 
            "You are an empathetic and highly intelligent AI pathology assistant. "
            "Analyze the image and answer questions conversationally, concisely, and accurately. "
            "Avoid unnecessarily long medical disclaimers if you have already stated them."
        )

        print("[System] AgentMemoryBot initialized successfully.", file=sys.stderr)

    # =========================
    # 对外聊天接口 (Stream API)
    # =========================
    def chat_stream(self, prompt: str, pil_img=None, streamer_callback=None, has_new_image=False):
        
        if has_new_image:
            self.history = []
            self.memory_summary = ""
            self.first_image_message = None
            if streamer_callback:
                streamer_callback("\n[System: New ROI detected. Context cleared. Starting fresh...]\n\n")
        if len(self.history) == 0 and pil_img is not None:
            user_message = self._build_user_message(prompt, pil_img)
            self.first_image_message = user_message
        else:
            user_message = self._build_user_message(prompt, None)

        self.history.append(user_message)

        before_tokens = self._estimate_text_tokens(self._build_model_messages())
        if before_tokens > self.max_input_tokens and streamer_callback:
            streamer_callback("\n[System: Generating memory summary to save VRAM...]\n\n")
            
        self._prune_history()

        messages_for_model = self._build_model_messages()
        reply = self._generate_response_stream(messages_for_model, streamer_callback)

        self.history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": reply}],
        })

        return reply, self.history

    def _build_user_message(self, text: str, pil_img=None) -> Dict[str, Any]:
        content = []
        if pil_img is not None:
            content.append({"type": "image", "image": pil_img})
        content.append({"type": "text", "text": text})
        return {"role": "user", "content": content}

    def _build_model_messages(self) -> List[Dict[str, Any]]:
        messages = [{"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}]

        if self.memory_summary.strip():
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": f"[Memory Summary] {self.memory_summary}"}],
            })

        if self.first_image_message is not None:
            messages.append(self.first_image_message)

        for msg in self.history:
            if self.first_image_message is not None and msg == self.first_image_message:
                continue
            messages.append(msg)

        return messages

    def _prune_history(self):
        messages = self._build_model_messages()
        token_count = self._estimate_text_tokens(messages)

        if token_count <= self.max_input_tokens:
            return

        recent_msgs = self.history[-self.recent_message_budget:]
        
        old_msgs = []
        for msg in self.history:
            if self.first_image_message is not None and msg == self.first_image_message:
                continue
            if msg in recent_msgs:
                continue
            old_msgs.append(msg)

        if old_msgs:
            new_summary = self._make_summary_from_messages(old_msgs)
            self.memory_summary = self._merge_summaries(self.memory_summary, new_summary)

        new_history = []
        if self.first_image_message is not None:
            new_history.append(self.first_image_message)

        for msg in recent_msgs:
            if msg != self.first_image_message and msg not in new_history:
                new_history.append(msg)

        self.history = new_history
        self._hard_prune_if_needed()

    def _hard_prune_if_needed(self):
        while True:
            messages = self._build_model_messages()
            if self._estimate_text_tokens(messages) <= self.max_input_tokens:
                break

            removable_indices = [
                i for i, msg in enumerate(self.history) 
                if not (self.first_image_message is not None and msg == self.first_image_message)
            ]

            if removable_indices:
                self.history.pop(removable_indices[0])
                continue

            if len(self.memory_summary) > 100:
                self.memory_summary = "..." + self.memory_summary[-100:]
                continue
            break

    def _make_summary_from_messages(self, messages: List[Dict[str, Any]]) -> str:
        summary = self._summarize_with_model(messages)
        if not summary.strip():
            summary = self._make_fallback_summary(messages)

        if len(summary) > self.summary_char_limit:
            summary = summary[:self.summary_char_limit]
        return summary

    def _summarize_with_model(self, messages: List[Dict[str, Any]]) -> str:
        dialogue_text = " ".join([
            f"{m.get('role', 'unknown')}: {' '.join([i.get('text', '') for i in m.get('content', []) if i.get('type') == 'text'])}"
            for m in messages
        ]).strip()

        if not dialogue_text: return ""

        summary_messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Summarize the dialogue history into short memory notes. Keep only key medical facts and unresolved goals. Do not invent information. Be concise."}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": f"Please summarize this dialogue:\n\n{dialogue_text}"}],
            },
        ]

        try:
            text = self.processor.apply_chat_template(summary_messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], padding=True, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.summary_max_new_tokens,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )

            gen_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
            return self.processor.batch_decode(gen_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return ""
        except Exception:
            return ""

    def _make_fallback_summary(self, messages: List[Dict[str, Any]]) -> str:
        parts = []
        for m in messages:
            text = " ".join([i.get("text", "") for i in m.get("content", []) if i.get("type") == "text"])
            if text: parts.append(f"{m.get('role', 'unknown')}: {text}")
        
        summary = " | ".join(parts)
        return "..." + summary[-self.summary_char_limit:] if len(summary) > self.summary_char_limit else summary

    def _merge_summaries(self, old_sum: str, new_sum: str) -> str:
        if not old_sum: merged = new_sum
        elif not new_sum: merged = old_sum
        else: merged = old_sum + " | " + new_sum
        return "..." + merged[-self.summary_char_limit:] if len(merged) > self.summary_char_limit else merged

    def _estimate_text_tokens(self, messages: List[Dict[str, Any]]) -> int:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return len(self.processor.tokenizer.encode(text, add_special_tokens=False))

    def _generate_response_stream(self, messages: List[Dict[str, Any]], streamer_callback) -> str:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(self.model.device)

        streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            
            do_sample=True,          
            temperature=0.5,         
            top_p=0.9,               
            repetition_penalty=1.1   
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        raw_reply = ""
        for new_text in streamer:
                    raw_reply += new_text
                    if streamer_callback:
                        streamer_callback(new_text)

        thread.join()
        return raw_reply
    


# # llm_manager.py
# import os
# import json, torch
# from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig,AutoModelForCausalLM, TextIteratorStreamer
# import torchvision.transforms as T
# from torchvision.transforms.functional import InterpolationMode
# from lemon import model
# from utils import resource_path
# import tempfile
# from threading import Thread
# from contextlib import contextmanager
# from qwen_vl_utils import process_vision_info 
# #########################################################################################################
# #Loading LLM 
# #########################################################################################################
# #@lru_cache(maxsize=None) This allows the model stays in cache but will makes the software stuck if the user didn't have a large ARM GPU
# def load_llm(config_path: str, precision: str = "8bit"):
#     """
#     Return (model, tokenizer, cfg) — cached by config_path.
#     """
#     with open(resource_path(config_path), "r", encoding="utf-8") as fh:
#         cfg = json.load(fh)
        
#     # ---------- HuatuoGPT-Vision & Lingshu-7B ----------------
#     repo_lower = cfg["repo"].lower()
#     # ---------- HuatuoGPT‑Vision‑7B‑Qwen2.5VL & Lingshu-7B ---------------------------
#     if repo_lower.endswith("huatuogpt-vision-7b-qwen2.5vl") or repo_lower.endswith("lingshu-7b"):
#         from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
#         model_kwargs = {
#             "device_map": "auto",
#             "trust_remote_code": True,
#         }

#         if precision == "4bit":
#             # For 4-bit
#             model_kwargs["quantization_config"] = BitsAndBytesConfig(
#                 load_in_4bit=True,
#                 bnb_4bit_compute_dtype=torch.bfloat16,
#                 bnb_4bit_quant_type="nf4",
#                 bnb_4bit_use_double_quant=True,
#             )
#         elif precision == "16bit":
#             # For 16-bit
#             model_kwargs["torch_dtype"] = torch.bfloat16
#         else: # Default to 8bit
#             # For 8-bit
#             model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

#         model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#             cfg["repo"],
#             **model_kwargs,
#         ).eval()
        
#         tokenizer = AutoProcessor.from_pretrained(cfg["repo"])
#         return model, tokenizer, cfg


# #########################################################################################################
# # Chat with LLM
# #########################################################################################################
# # 提示：之前的 _compress_memory 函数可以直接整块删除了，不再需要。

# def stream_reply(model, tokenizer, llm_config, pil_img, prompt, history, streamer_callback=None, has_new_image=False):
#     """
#     Returns (assistant_reply_text, updated_history)
#     """
#     repo_lower = llm_config["repo"].lower()
    
#     if repo_lower.endswith("huatuogpt-vision-7b-qwen2.5vl") or repo_lower.endswith("lingshu-7b"):
#         MAX_TURNS = 5 

#         # ==========================================
#         # 1. 换图清空上下文逻辑
#         # ==========================================
#         if has_new_image:
#             history.clear() # 🚀 新图进来，彻底清空之前的对话
#             if streamer_callback: # <--- 必须缩进在 if has_new_image 的里面！
#                 streamer_callback("\n<i style='color:gray;'>[System: Vision updated, previous context cleared.]</i><br>\n")

#         # ==========================================
#         # 2. 构建增量 History (Visual Anchoring)
#         # ==========================================
#         if len(history) == 0:
#             history.append({
#                 "role": "user",
#                 "content": [
#                     {"type": "image", "image": pil_img}, # 必须在第一轮塞入图片
#                     {"type": "text",  "text": prompt},
#                 ],
#             })
#         else:
#             history.append({
#                 "role": "user",
#                 "content": [{"type": "text", "text": prompt}],
#             })

#         # ==========================================
#         # 3. 滚动截断记忆 (保留首图 + 最近的 N-1 轮)
#         # ==========================================
#         # 一轮对话 = 1 user + 1 assistant (2个 message)。
#         # 当 MAX_TURNS=5 时，发送给模型前最多允许 9 条记录 (第一轮 user + 中间 3 轮完整对话 (6条) + 最新的 1 个 user = 8条。不，是 4轮完整的 = 8 + 最新 1条 = 9条)
#         # 如果长度超出，我们永远保留 history[0] (带图的那条)，并只取最后面的 8 条。
#         if len(history) > (MAX_TURNS * 2) - 1:
#             # 🚀 截断历史，保留带有 image 的首轮，并拼接上最近的 4 轮对话
#             history = [history[0]] + history[-(MAX_TURNS * 2 - 2):]

#         # ==========================================
#         # 4. 准备模型推理
#         # ==========================================
#         text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
#         image_inputs, video_inputs = process_vision_info(history)

#         inputs = tokenizer(
#             text=[text],
#             images=image_inputs,
#             videos=video_inputs,
#             padding=True,
#             return_tensors="pt",
#         ).to(model.device)

#         streamer = TextIteratorStreamer(tokenizer.tokenizer, skip_prompt=True, skip_special_tokens=True)

#         generation_kwargs = dict(
#             **inputs,
#             streamer=streamer,
#             max_new_tokens=256,
#             eos_token_id=tokenizer.tokenizer.eos_token_id,
            
#             # 🚀 [核心修复：打破复读机魔咒]
#             do_sample=True,          # 开启采样（不再是死板的贪婪匹配）
#             temperature=0.5,         # 温度适中：0.5既能保证医学严谨性，又不会死板
#             top_p=0.9,               # 核心采样：避免胡言乱语
#             repetition_penalty=1.1   # 重复惩罚：强烈阻止它连续输出一模一样的句子！
#         )
        
#         txt = ""
#         thread = Thread(target=model.generate, kwargs=generation_kwargs)
#         thread.start()

#         # 实时推流给前端
#         for new_text in streamer:
#             formatted_text = new_text.replace("\n", "<br>")
#             if streamer_callback:
#                 streamer_callback(formatted_text)
#             txt += formatted_text
        
#         thread.join() 
        
#         # ==========================================
#         # 5. 保存助手回复并返回
#         # ==========================================
#         history.append({
#             "role": "assistant",
#             "content": [{"type": "text", "text": txt}]
#         })
        
#         return txt, history