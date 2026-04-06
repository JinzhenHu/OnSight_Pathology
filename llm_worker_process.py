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
    from llm_manager import AgentMemoryBot
    with Image.open(args.image) as img:
        image_pil = img.convert("RGB")

    # 🚀 1. 直接实例化我们的记忆管家
    bot = AgentMemoryBot(config_path=args.cfg, precision=args.precision)
    
    # 初始化成功，给前端发送 Ready 信号
    print(json.dumps({"type": "ready"}), flush=True)

except Exception as e:
    # 🚨 核心修复：如果模型加载崩溃（如 OOM、代码错误），将报错详情直接发给前端 UI！
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