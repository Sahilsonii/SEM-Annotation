"""
worker.py -- YOLO inference subprocess
=======================================
Launched by main.py as a child process.
Protocol (stdin/stdout binary, length-prefixed):
  - Sends "READY" (5 bytes) when model is loaded.
  - Reads: 4-byte big-endian uint32 length, then <length> bytes of raw image data.
  - Writes: 4-byte big-endian uint32 length, then pickle of result dict.
"""

import sys
import io
import struct
import pickle
import traceback


def load_and_serve(weights_path: str):
    print(f"[worker] Loading YOLO from: {weights_path}", file=sys.stderr, flush=True)

    # ----------------------------------------------------------------
    # Load torch & ultralytics — this can take 3-10 min on first run
    # due to Windows Defender scanning DLLs. We just wait it out.
    # ----------------------------------------------------------------
    print("[worker] Importing torch (may take several minutes on first run)...", file=sys.stderr, flush=True)
    import torch
    print(f"[worker] torch {torch.__version__}, CUDA={torch.cuda.is_available()}", file=sys.stderr, flush=True)

    print("[worker] Importing ultralytics YOLO...", file=sys.stderr, flush=True)
    from ultralytics import YOLO
    import numpy as np
    from PIL import Image

    print("[worker] Loading model weights...", file=sys.stderr, flush=True)
    model = YOLO(weights_path)

    # Warm-up pass
    print("[worker] Warming up model (224x224 dummy)...", file=sys.stderr, flush=True)
    dummy = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    model(dummy, imgsz=224, conf=0.25, verbose=False)

    print("[worker] Model ready!", file=sys.stderr, flush=True)

    # Signal ready to parent
    sys.stdout.buffer.write(b"READY")
    sys.stdout.buffer.flush()

    # Serve inference requests
    while True:
        try:
            # Read request length
            raw_len = sys.stdin.buffer.read(4)
            if not raw_len or len(raw_len) < 4:
                print("[worker] stdin closed, exiting.", file=sys.stderr, flush=True)
                break

            req_len = struct.unpack(">I", raw_len)[0]
            img_bytes = sys.stdin.buffer.read(req_len)

            # Run inference — auto-resize any input resolution to model imgsz
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            results = model(img, imgsz=224, conf=0.05, verbose=False)
            boxes_raw = results[0].boxes

            boxes = []
            for box in boxes_raw:
                boxes.append({
                    "cls_id": int(box.cls[0].item()),
                    "conf":   float(box.conf[0].item()),
                    "xyxy":   [float(c) for c in box.xyxy[0].tolist()],
                })

            result = {"boxes": boxes}
            print(f"[worker] Inference done: {len(boxes)} boxes", file=sys.stderr, flush=True)

        except Exception as e:
            print(f"[worker] Inference error: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            result = {"error": str(e)}

        # Write response
        resp_bytes = pickle.dumps(result)
        sys.stdout.buffer.write(struct.pack(">I", len(resp_bytes)))
        sys.stdout.buffer.write(resp_bytes)
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[worker] ERROR: no weights path given", file=sys.stderr, flush=True)
        sys.exit(1)
    load_and_serve(sys.argv[1])
