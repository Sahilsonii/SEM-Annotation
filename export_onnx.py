"""
export_onnx.py
==============
Exports the best YOLO11m weights to ONNX format.
ONNX loads in ~1 second vs 3-10 minutes for full PyTorch CUDA.
Run once: python export_onnx.py
"""
from pathlib import Path
print("Importing torch + ultralytics...")
from ultralytics import YOLO

weights = Path(__file__).parent / "src/runs/detect/yolo11m_20260420_205814/weights/best.pt"
out_dir = weights.parent

print(f"Loading {weights}...")
model = YOLO(str(weights))

print("Exporting to ONNX (imgsz=224)...")
# dynamic=False gives a fixed 224x224 input — fastest inference
path = model.export(format="onnx", imgsz=224, dynamic=False, simplify=True)
print(f"\nDone! ONNX saved to: {path}")
