# llm_manager.py — unified for Windows (CUDA) and macOS (MPS/CPU)
import os
import sys

# ---------------------------------------------------------------
# OnSight-owned HF cache path.
# Must be set BEFORE importing transformers/huggingface_hub so the
# constants module captures it. Same parent as utils._get_weights_path
# so all OnSight model files live under one folder, which keeps
# Windows Defender / corporate AV happy.
# ---------------------------------------------------------------
def _onsight_hf_cache_dir() -> str:
    if sys.platform == "darwin":
        return os.path.join(
            os.path.expanduser("~"), "Library", "Application Support",
            "OnSightPathology", "hf_cache", "hub",
        )
    if sys.platform == "win32":
        return os.path.join(
            os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
            "OnSightPathology", "hf_cache", "hub",
        )
    return os.path.join(
        os.path.expanduser("~"), ".cache",
        "OnSightPathology", "hf_cache", "hub",
    )

_ONSIGHT_HF_CACHE = _onsight_hf_cache_dir()
try:
    os.makedirs(_ONSIGHT_HF_CACHE, exist_ok=True)
except Exception:
    pass
os.environ.setdefault("HF_HUB_CACHE", _ONSIGHT_HF_CACHE)
os.environ.setdefault("HF_HOME", os.path.dirname(_ONSIGHT_HF_CACHE))

# Heavy imports must come AFTER the env vars above.
import json
import torch
from threading import Thread
from typing import List, Dict, Any, Optional, Tuple

from transformers import (
    BitsAndBytesConfig,
    TextIteratorStreamer,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info
from utils import resource_path


# ---------------------------------------------------------------
# Download-progress hook.
# Patches every tqdm reference already imported under huggingface_hub
# and transformers, so both the "Fetching N files" outer bar and the
# inner byte-level bars route through our progress_cb.
# Used as a context manager so patches are reverted on exit.
# ---------------------------------------------------------------
class _HfProgressHook:
    def __init__(self, progress_cb, pct_start=10, pct_end=85):
        self._cb = progress_cb
        self._lo = pct_start
        self._hi = pct_end
        self._total = 0
        self._done = 0
        self._last_pct = -1
        self._last_mb = -1.0
        self._patches = []

    def __enter__(self):
        hook = self

        class _HookedTqdm:
                    def __init__(self, iterable=None, *args, **kwargs):
                        # tqdm signature is tqdm(iterable=None, ...), so the first
                        # positional arg may be the wrapped iterable. Capture it so
                        # `for x in tqdm(items):` still works.
                        self._iterable = iterable
                        self.total = kwargs.get("total")
                        if self.total is None and iterable is not None:
                            try:
                                self.total = len(iterable)
                            except (TypeError, AttributeError):
                                self.total = 0
                        self.total = self.total or 0
                        self.n = 0
                        unit = kwargs.get("unit", "")
                        self.is_bytes = (unit in ("B", "iB")
                                        or bool(kwargs.get("unit_scale")))
                        if self.total > 0 and self.is_bytes:
                            hook._total += self.total

                    def __iter__(self):
                        if self._iterable is None:
                            return iter(())
                        for item in self._iterable:
                            self.n += 1
                            yield item

                    def update(self, increment=1):
                        self.n += increment
                        if self.total > 0 and self.is_bytes:
                            hook._done += increment
                            hook._maybe_report()

                    def reset(self, total=None):
                        if total is not None and total > 0 and self.is_bytes:
                            delta = total - self.total if self.total > 0 else total
                            hook._total += delta
                            self.total = total
                        self.n = 0

                    def close(self): pass
                    def __enter__(self): return self
                    def __exit__(self, *a): pass
                    def set_description(self, *a, **kw): pass
                    def set_postfix(self, *a, **kw): pass
                    def refresh(self): pass
                    def write(self, *a, **kw): pass
                    def display(self, *a, **kw): pass
                    def clear(self, *a, **kw): pass
                    disable = False
        # Walk every already-loaded module under HF / transformers / tqdm
        # and replace its tqdm attribute. This catches local re-imports
        # like `from .utils import tqdm` that won't follow a parent patch.
        seen = set()
        for mod_name, mod in list(sys.modules.items()):
            if mod is None:
                continue
            if not (mod_name.startswith("huggingface_hub")
                    or mod_name.startswith("transformers")
                    or mod_name in ("tqdm", "tqdm.auto", "tqdm.std")):
                continue
            try:
                original = getattr(mod, "tqdm", None)
            except Exception:
                continue
            if original is None or id(original) in seen:
                continue
            try:
                setattr(mod, "tqdm", _HookedTqdm)
                self._patches.append((mod, "tqdm", original))
                seen.add(id(original))
            except Exception:
                pass

        # Announce that we're starting, in case the first download takes
        # a while before tqdm fires its first update().
        try:
            self._cb(self._lo, "Connecting to HuggingFace...")
        except Exception:
            pass
        return self

    def __exit__(self, *exc):
        for mod, attr, original in self._patches:
            try:
                setattr(mod, attr, original)
            except Exception:
                pass
        self._patches.clear()
        return False

    def _maybe_report(self):
        if self._total <= 0:
            return
        frac = min(1.0, self._done / self._total)
        pct = int(self._lo + frac * (self._hi - self._lo))
        mb_done = self._done / (1024 * 1024)
        if pct == self._last_pct and abs(mb_done - self._last_mb) < 5.0:
            return
        self._last_pct = pct
        self._last_mb = mb_done
        mb_total = self._total / (1024 * 1024)
        try:
            self._cb(pct, f"Downloading model weights — {mb_done:.0f} / {mb_total:.0f} MB")
        except Exception:
            pass


# ----------------------------------------------------------
# Platform / device detection (done once at import time).
# Windows builds are expected to ship with CUDA; macOS builds
# use MPS (Apple Silicon) or CPU fallback.
# ----------------------------------------------------------
# ----------------------------------------------------------
# Platform / device detection (done once at import time).
# ----------------------------------------------------------
_IS_MAC = sys.platform == "darwin"

# bitsandbytes 4-bit / 8-bit CUDA kernels require compute capability >= 7.5
_BNB_SUPPORTED = False
_GPU_VRAM_GB = 0.0
_GPU_CC = (0, 0)
_GPU_NAME = ""

if torch.cuda.is_available():
    _DEVICE = "cuda"
    try:
        _GPU_CC = torch.cuda.get_device_capability(0)
        _GPU_VRAM_GB = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        _GPU_NAME = torch.cuda.get_device_name(0)
        _BNB_SUPPORTED = _GPU_CC >= (7, 5)
    except Exception:
        _BNB_SUPPORTED = False
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    _DEVICE = "mps"
else:
    _DEVICE = "cpu"

class AgentMemoryBot:
    """
    Agent memory management
    - Maintains a dialogue history with roles (user/assistant) and content (text/image).
    - Implements a memory summarization mechanism to condense old dialogue into a short
      summary when token limits are exceeded.
    - Provides a streaming chat interface that can handle new images and prompts, and
      generates responses incrementally.
    """
    def __init__(self, config_path: str, precision: str = "16bit", progress_cb=None):
        # progress_cb(percent:int, text:str) — optional UI progress callback.
        self._pcb = progress_cb if callable(progress_cb) else (lambda p, t: None)
        self._pcb(5, "Loading config...")

        with open(resource_path(config_path), "r", encoding="utf-8") as fh:
            self.cfg = json.load(fh)

        model_id = self.cfg["repo"]
        print(f"[System] Loading model: {model_id} on {_DEVICE}", file=sys.stderr)

        # bitsandbytes (4/8-bit) needs CUDA + compute capability >= 7.5.
        # On Mac, on CPU, or on old NVIDIA cards (Pascal/Volta), gracefully
        # downgrade so we don't crash with "named symbol not found".
        if precision in ("4bit", "8bit") and not _BNB_SUPPORTED:
            if _DEVICE == "cuda":
                # Old CUDA card (e.g. GTX 1080 Ti, CC 6.1).
                # 16-bit Lingshu-7B needs ~14 GB; if VRAM is below that
                # threshold, fall back to CPU instead of OOM-crashing.
                if _GPU_VRAM_GB < 14.0:
                    print(
                        f"[System] GPU ({_GPU_NAME}, CC {_GPU_CC[0]}.{_GPU_CC[1]}, "
                        f"{_GPU_VRAM_GB:.1f} GB VRAM) is too old for 4/8-bit "
                        f"quantization and too small for 16-bit. "
                        f"Falling back to CPU — responses will be slow.",
                        file=sys.stderr,
                    )
                    # Re-route to CPU. Mutate the module-level constant only
                    # for this load by switching the local one we pass to .to().
                    self._effective_device = "cpu"
                    precision = "16bit"
                else:
                    print(
                        f"[System] GPU ({_GPU_NAME}, CC {_GPU_CC[0]}.{_GPU_CC[1]}) "
                        f"doesn't support bnb quantization. Using 16-bit on CUDA.",
                        file=sys.stderr,
                    )
                    self._effective_device = "cuda"
                    precision = "16bit"
            else:
                print(
                    f"[System] {precision} requires CUDA; "
                    f"falling back to 16-bit on {_DEVICE}.",
                    file=sys.stderr,
                )
                self._effective_device = _DEVICE
                precision = "16bit"
        else:
            self._effective_device = _DEVICE

        self._pcb(10, "Loading tokenizer / processor...")

        # Wrap both from_pretrained calls so download progress streams to the UI.
        # cache_dir is explicit (in addition to HF_HUB_CACHE env var) to make
        # sure weights always land in OnSight's own folder, not in ~/.cache.
        with _HfProgressHook(self._pcb, pct_start=10, pct_end=85):
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                cache_dir=_ONSIGHT_HF_CACHE,
            )

            # ------- Build model_kwargs based on platform / precision -------
            model_kwargs: Dict[str, Any] = {"trust_remote_code": True}

            if self._effective_device == "cuda":
                # Windows / CUDA path: rely on accelerate to place layers,
                # use SDPA attention for speed.
                model_kwargs["device_map"] = "auto"
                model_kwargs["attn_implementation"] = "sdpa"
            else:
                # macOS (MPS / CPU): no device_map; we will .to(device) after load.
                # MPS lacks some SDPA kernels, so use eager attention there.
                model_kwargs["attn_implementation"] = "eager" if self._effective_device == "mps" else "sdpa"

            if precision == "4bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                )
            elif precision == "16bit":
                model_kwargs["torch_dtype"] = torch.bfloat16
            else:  # 8bit
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                cache_dir=_ONSIGHT_HF_CACHE,
                **model_kwargs,
            ).eval()

        # On Mac we need to manually move 16bit weights to MPS/CPU.
        # On CUDA, accelerate has already placed them via device_map.
        if precision == "16bit" and self._effective_device != "cuda":
            self._pcb(90, f"Moving model to {self._effective_device.upper()}...")
            self.model.to(self._effective_device)
        else:
            self._pcb(90, "Finalizing model placement...")

        self._pcb(96, "Initializing memory module...")

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

        self._pcb(99, "Finalizing...")
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

        if not dialogue_text:
            return ""

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
            if text:
                parts.append(f"{m.get('role', 'unknown')}: {text}")

        summary = " | ".join(parts)
        return "..." + summary[-self.summary_char_limit:] if len(summary) > self.summary_char_limit else summary

    def _merge_summaries(self, old_sum: str, new_sum: str) -> str:
        if not old_sum:
            merged = new_sum
        elif not new_sum:
            merged = old_sum
        else:
            merged = old_sum + " | " + new_sum
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

        # Sampling defaults differ historically between platforms; preserve the
        # original behaviour so reproducibility of the published evaluation is
        # unchanged. CUDA path uses light sampling; MPS/CPU prefers greedy
        # decoding (more stable, avoids occasional MPS softmax NaNs).
        if self._effective_device == "cuda":
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.5,
                top_p=0.9,
                repetition_penalty=1.1,
            )
        else:
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                do_sample=False,
                repetition_penalty=1.1,
            )

        def generate_with_error_handling():
            try:
                self.model.generate(**generation_kwargs)
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                if streamer_callback:
                    streamer_callback(
                        f"<br><br><b style='color:#FA5252;'>[Model Crash inside Thread]:</b> {str(e)}<br>"
                    )
                    print(f"\n[Generation Error Details]:\n{error_trace}", file=sys.stderr)

                # Unblock the streamer iterator so the parent doesn't hang.
                if hasattr(streamer, "text_queue") and hasattr(streamer, "stop_signal"):
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