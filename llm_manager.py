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
    Agent memory management 
    - Maintains a dialogue history with roles (user/assistant) and content (text/image
    - Implements a memory summarization mechanism to condense old dialogue into a short summary when token limits are exceeded.
    - Provides a streaming chat interface that can handle new images and prompts, and generates responses incrementally.
    """
    def __init__(self, config_path: str, precision: str = "16bit"):
        with open(resource_path(config_path), "r", encoding="utf-8") as fh:
            self.cfg = json.load(fh)

        model_id = self.cfg["repo"]
        print(f"[System] Loading model: {model_id}", file=sys.stderr)

        self.processor = AutoProcessor.from_pretrained(model_id)
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        model_kwargs = {
            # "device_map": "auto",
            "trust_remote_code": True,
            "attn_implementation": "eager" if device == "mps" else "sdpa"
        }

        if precision == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True  # 🚀 关键修复：允许 4bit 模型在 CPU 上运行
            )
        elif precision == "16bit":
            model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, **model_kwargs
        ).eval()
        if precision == "16bit":
            self.model.to(device)
        # =========================
        # Memory Management Attributes
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
    # Core Chat Function with Streaming and Memory Management
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
            
            do_sample=False,          
            #temperature=0.5,         
            #top_p=0.9,               
            repetition_penalty=1.1   
        )

        def generate_with_error_handling():
            try:
                self.model.generate(**generation_kwargs)
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                if streamer_callback:
                    streamer_callback(f"<br><br><b style='color:#FA5252;'>[Model Crash inside Thread]:</b> {str(e)}<br>")
                    print(f"\n[Generation Error Details]:\n{error_trace}", file=sys.stderr)
                

                if hasattr(streamer, 'text_queue') and hasattr(streamer, 'stop_signal'):
                    streamer.text_queue.put(streamer.stop_signal)

        thread = Thread(target=generate_with_error_handling)
        thread.start()

        raw_reply = ""
        for new_text in streamer:
                    raw_reply += new_text
                    if streamer_callback:
                        streamer_callback(new_text)

        thread.join()
        return raw_reply
