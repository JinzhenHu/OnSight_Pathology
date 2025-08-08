import argparse
import json
import sys
from PIL import Image
from llm_manager import load_llm   
from llm_manager import stream_reply

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", required=True)
parser.add_argument("--image", required=True)
args = parser.parse_args()

with Image.open(args.image) as img:
    image_pil = img.convert("RGB")

model, tokenizer, cfg = load_llm(args.cfg)
msgs = []

print(json.dumps({"type": "ready"}), flush=True)

# Read prompts from stdin
while True:
    line = sys.stdin.readline()
    if not line:
        break
    try:
        msg = json.loads(line)
        if msg["type"] == "prompt":
            reply, msgs = stream_reply(model, tokenizer, cfg, image_pil, msg["text"], msgs)
            print(json.dumps({"type": "reply", "text": reply}), flush=True)
    except Exception as e:
        print(json.dumps({"type": "error", "text": str(e)}), flush=True)
