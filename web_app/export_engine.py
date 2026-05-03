import sys, time
print("Step 0: Script started", flush=True)
print(f"Python: {sys.version}", flush=True)

print("Step 1: Importing torch...", flush=True)
t0 = time.time()
import torch
print(f"  torch imported in {time.time()-t0:.1f}s, version={torch.__version__}, cuda={torch.cuda.is_available()}", flush=True)

print("Step 2: Importing ultralytics...", flush=True)
t1 = time.time()
from ultralytics import YOLO
print(f"  ultralytics imported in {time.time()-t1:.1f}s", flush=True)

WEIGHTS = r"C:\Users\asus\Downloads\SEM-Annotation\src\runs\detect\yolo11m_20260420_205814\weights\best.pt"

print(f"Step 3: Loading YOLO({WEIGHTS})...", flush=True)
t2 = time.time()
model = YOLO(WEIGHTS)
print(f"  Model loaded in {time.time()-t2:.1f}s", flush=True)

print("Step 4: Test inference on dummy 224x224...", flush=True)
import numpy as np
from PIL import Image
dummy = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
t3 = time.time()
results = model(dummy, imgsz=224, conf=0.25, verbose=False)
print(f"  Inference done in {time.time()-t3:.1f}s, {len(results[0].boxes)} boxes", flush=True)

print("Step 5: Trying TensorRT export...", flush=True)
t4 = time.time()
try:
    model.export(format="engine", imgsz=224, half=True)
    print(f"  TensorRT export done in {time.time()-t4:.1f}s", flush=True)
except Exception as e:
    print(f"  TensorRT FAILED: {e}", flush=True)
    print("  Trying ONNX export...", flush=True)
    try:
        model.export(format="onnx", imgsz=224)
        print(f"  ONNX export done in {time.time()-t4:.1f}s", flush=True)
    except Exception as e2:
        print(f"  ONNX also FAILED: {e2}", flush=True)

print(f"\nTotal: {time.time()-t0:.1f}s", flush=True)
