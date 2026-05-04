import os
import torch
from typing import List, Dict, Any, Optional, Tuple

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info

class AgentMemoryBot:
    """
    一个简单的单图记忆 Bot

    设计目标：
    1. 第一张图永远保留
    2. 最近几条消息优先保留
    3. 更早的历史先用模型总结，再压缩成短摘要
    4. 如果模型总结失败，就退回普通字符串摘要
    """

    def __init__(
        self,
        model_id: str = "lingshu-medical-mllm/Lingshu-7B",
        precision: str = "4bit",
        max_input_tokens: int = 1200,
        max_new_tokens: int = 256,
        recent_message_budget: int = 4,
        summary_char_limit: int = 400,
        summary_max_new_tokens: int = 128,
        system_prompt: str = "You are a helpful multimodal assistant. Be concise and accurate.",
    ):
        print(f"[System] Loading model: {model_id}")

        self.processor = AutoProcessor.from_pretrained(model_id)

        quantization_config = None
        torch_dtype = torch.float16
        model_kwargs = {
            "device_map": "auto",  # 允许它自动切分
            "trust_remote_code": True,
        }

        if precision == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True  # 🚀 关键修复：允许 4bit 模型在 CPU 上运行
            )
        elif precision == "fp16":
            quantization_config = None
            torch_dtype = torch.float16
        elif precision == "bf16":
            quantization_config = None
            torch_dtype = torch.bfloat16
        else:
            raise ValueError("precision must be one of: '4bit', 'fp16', 'bf16'")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            device_map="auto",
        )

        self.history: List[Dict[str, Any]] = []
        self.memory_summary: str = ""
        self.first_image_message: Optional[Dict[str, Any]] = None

        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.summary_max_new_tokens = summary_max_new_tokens
        self.recent_message_budget = recent_message_budget
        self.summary_char_limit = summary_char_limit
        self.system_prompt = system_prompt

        print("[System] Bot initialized.")

if __name__ == "__main__":
        bot = AgentMemoryBot(
        model_id="lingshu-medical-mllm/Lingshu-7B",
        precision="4bit",
        max_input_tokens=300,
        max_new_tokens=256,
        summary_max_new_tokens=96,
        recent_message_budget=2,
        summary_char_limit=300,
        system_prompt="You are a pathology assistant. Be brief and accurate.",
    )
        print("Bot loaded successfully.")
