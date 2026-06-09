import os
import sys
import json
import argparse
import traceback
from PIL import Image

# ONLY FOR CPU EXE
if os.environ.get("BUILD_TYPE", "").upper() == "CPU":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TORCH_CUDA_DUMMY_DEVICE"] = "1"


# ==========================================================
# Progress reporting helper. Emits one JSON line to stdout.
# Works on both Windows and macOS — the parent process reads
# these structured lines and updates the loading UI.
# ==========================================================
def _emit_json(payload):
    try:
        print(json.dumps(payload), flush=True)
    except Exception:
        pass


# ==========================================================
# This is platform-independent; transformers and huggingface
# always use tqdm regardless of OS.
# ==========================================================
try:
    import tqdm as _tqdm_mod
    _OrigTqdm = _tqdm_mod.tqdm

    class _ProgressTqdm(_OrigTqdm):
        def display(self, msg=None, pos=None):
            try:
                total = self.total or 0
                pct = int(self.n * 100 / total) if total > 0 else -1
                desc = (self.desc or "").strip() or "Loading"
                last = getattr(self, "_last_emit_pct", -10)
                # Throttle: only emit when pct advanced >=2 or finished
                if pct < 0 or pct - last >= 2 or (total > 0 and self.n >= total):
                    self._last_emit_pct = pct
                    _emit_json({
                        "type": "progress",
                        "stage": "hf",
                        "percent": pct,
                        "text": (f"{desc} ({pct}%)" if pct >= 0 else desc),
                    })
            except Exception:
                pass

        def close(self):
            try:
                if (self.total or 0) > 0:
                    _emit_json({
                        "type": "progress",
                        "stage": "hf",
                        "percent": 100,
                        "text": f"{(self.desc or 'Loading').strip()} (done)",
                    })
            except Exception:
                pass
            super().close()

    _tqdm_mod.tqdm = _ProgressTqdm
    try:
        import tqdm.auto as _tqdm_auto
        _tqdm_auto.tqdm = _ProgressTqdm
    except Exception:
        pass
except Exception:
    pass


parser = argparse.ArgumentParser()
parser.add_argument("--cfg", required=True)
parser.add_argument("--image", required=True)
parser.add_argument(
    "--precision",
    choices=["4bit", "8bit", "16bit"],
    default="16bit",
    help="Model loading precision."
)
args = parser.parse_args()

try:
    _emit_json({"type": "progress", "stage": "init", "percent": 2,
                "text": "Initializing AI engine..."})

    from llm_manager import AgentMemoryBot

    with Image.open(args.image) as img:
        image_pil = img.convert("RGB")

    def _progress(percent, text):
        _emit_json({"type": "progress", "stage": "model",
                    "percent": int(percent), "text": str(text)})

    # load model in the worker process (now with progress callback)
    bot = AgentMemoryBot(
        config_path=args.cfg,
        precision=args.precision,
        progress_cb=_progress,
    )

    _emit_json({"type": "progress", "stage": "done", "percent": 100,
                "text": "Model ready"})
    print(json.dumps({"type": "ready"}), flush=True)

except Exception as e:
    error_msg = f"Model Initialization Failed:\n{traceback.format_exc()}"
    print(json.dumps({"type": "error", "text": error_msg}), flush=True)
    sys.exit(1)


def streamer_callback(chunk):
    """Takes a text chunk, wraps it in JSON, and prints it to stdout."""
    print(json.dumps({"type": "chunk", "text": chunk}), flush=True)


has_new_image = False

while True:
    line = sys.stdin.readline()
    if not line:
        break
    try:
        msg = json.loads(line)

        if msg["type"] == "update_image":
            with Image.open(msg["path"]) as img:
                image_pil = img.convert("RGB")
            has_new_image = True

        elif msg["type"] == "prompt":
            reply, _ = bot.chat_stream(
                prompt=msg["text"],
                pil_img=image_pil,
                streamer_callback=streamer_callback,
                has_new_image=has_new_image
            )
            has_new_image = False
            print(json.dumps({"type": "stream_end"}), flush=True)

    except Exception as e:
        error_msg = f"Runtime Error:\n{traceback.format_exc()}"
        print(json.dumps({"type": "error", "text": error_msg}), flush=True)