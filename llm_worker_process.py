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


while True:
    line = sys.stdin.readline()
    if not line:
        break
    try:
        msg = json.loads(line)
        if msg["type"] == "prompt":
            reply, msgs = stream_reply(
                model, tokenizer, cfg,
                image_pil, msg["text"], msgs,
                streamer_callback=streamer_callback
            )
            print(json.dumps({"type": "stream_end"}), flush=True)
    except Exception as e:
        print(json.dumps({"type": "error", "text": str(e)}), flush=True)