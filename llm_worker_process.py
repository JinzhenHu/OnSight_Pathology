import os

# ONLY FOR CPU EXE
if os.environ.get("BUILD_TYPE", "").upper() == "CPU":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TORCH_CUDA_DUMMY_DEVICE"] = "1"

import argparse
import json
import sys
from PIL import Image
from llm_manager import load_llm
from llm_manager import stream_reply

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", required=True)
parser.add_argument("--image", required=True)
parser.add_argument(
    "--precision", 
    choices=["4bit", "8bit", "16bit"], 
    default="8bit",
    help="Model loading precision."
)
args = parser.parse_args()

with Image.open(args.image) as img:
    image_pil = img.convert("RGB")

model, tokenizer, cfg = load_llm(args.cfg, precision=args.precision)
msgs = []
print(json.dumps({"type": "ready"}), flush=True)

def streamer_callback(chunk):
    """Takes a text chunk, wraps it in JSON, and prints it to stdout."""
    print(json.dumps({"type": "chunk", "text": chunk}), flush=True)

has_new_image = False  # 🚀 [新增状态位] 记录用户是否中途换了图

while True:
    line = sys.stdin.readline()
    if not line:
        break
    try:
        msg = json.loads(line)
        
        # 🚀 [新增逻辑] 接收到换图信号
        if msg["type"] == "update_image":
            with Image.open(msg["path"]) as img:
                image_pil = img.convert("RGB")
            has_new_image = True # 标记为 True，等下一个 prompt 来了带进去
            
        elif msg["type"] == "prompt":
            reply, msgs = stream_reply(
                model, tokenizer, cfg,
                image_pil, msg["text"], msgs,
                streamer_callback=streamer_callback,
                has_new_image=has_new_image # 🚀 传给大模型
            )
            has_new_image = False # 🚀 消费掉这个标记
            print(json.dumps({"type": "stream_end"}), flush=True)
            
    except Exception as e:
        print(json.dumps({"type": "error", "text": str(e)}), flush=True)
# while True:
#     line = sys.stdin.readline()
#     if not line:
#         break
#     try:
#         msg = json.loads(line)
#         if msg["type"] == "prompt":
#             reply, msgs = stream_reply(
#                 model, tokenizer, cfg,
#                 image_pil, msg["text"], msgs,
#                 streamer_callback=streamer_callback
#             )
#             print(json.dumps({"type": "stream_end"}), flush=True)
#     except Exception as e:
#         print(json.dumps({"type": "error", "text": str(e)}), flush=True)