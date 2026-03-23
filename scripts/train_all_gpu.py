"""
train_all_gpu.py

Script for headless sequential training of all 6 required YOLO models
on a dedicated GPU machine.

Run this script: python train_all_gpu.py
"""
import os
import sys

# ── CONFIG ────────────────────────────────────────────────────────────────────
EPOCHS   = 200
IMGSZ    = 1024
BATCH    = 4        # Increase to 8 or 16 if your GPU handles it
PATIENCE = 0        # 0 = disabled (run all epochs)
CACHE    = True     # Caches images to RAM for speed
AUGMENT  = True
COS_LR   = True

MODELS_TO_TRAIN = [
    # Small Models
    {"label": "yolov8s",  "weights": "yolov8s.pt"},
    {"label": "yolo26s",  "weights": "yolo26s.pt"},
    # Medium Models
    {"label": "yolov8m",  "weights": "yolov8m.pt"},
    {"label": "yolo26m",  "weights": "yolo26m.pt"},
    # Large Models
    {"label": "yolov8l",  "weights": "yolov8l.pt"},
    {"label": "yolo26l",  "weights": "yolo26l.pt"},
]
# ─────────────────────────────────────────────────────────────────────────────

HERE         = os.path.dirname(os.path.abspath(__file__))
BALANCED_DIR = os.path.join(HERE, "balanced_dataset")
YAML_PATH    = os.path.join(BALANCED_DIR, "data.yaml")

if not os.path.exists(YAML_PATH):
    sys.exit(f"ERROR: data.yaml not found at {YAML_PATH}\n"
             "Ensure the balanced dataset is present before training.")

# Validate and ensure data.yaml is correct
import yaml as _yaml
classes = {0: "PbI2", 1: "3D_pinholes", 2: "3D-2D_pinholes",
           3: "3D_background", 4: "3D-2D_background"}
names   = [classes[k] for k in sorted(classes)]
yaml_data = {
    "path": BALANCED_DIR, 
    "train": "images", 
    "val": "images",
    "nc": len(classes), 
    "names": names
}
with open(YAML_PATH, "w") as f:
    _yaml.dump(yaml_data, f, default_flow_style=False)
print(f"✓ data.yaml validated at: {YAML_PATH}")

# Import ModelHandler for consistent metrics saving
sys.path.insert(0, HERE)
try:
    from model_handler import ModelHandler
except ImportError:
    sys.exit("ERROR: Could not import ModelHandler. Ensure you run this from the sem_app directory.")

print(f"\n{'='*60}")
print(f"  GPU Lab Batch Training: {len(MODELS_TO_TRAIN)} models × {EPOCHS} epochs")
print(f"{'='*60}\n")

results_summary = []

for i, m in enumerate(MODELS_TO_TRAIN, 1):
    label   = m["label"]
    weights = m["weights"]
    wpath   = os.path.join(HERE, weights)

    print(f"\n[{i}/{len(MODELS_TO_TRAIN)}] ▶ Training {label}  (pre-trained weights: {weights})")
    print("-" * 60)

    try:
        # Load weights from disk if present, else auto-download
        handler = ModelHandler(wpath if os.path.exists(wpath) else weights)
        results, metrics_path = handler.train_model(
            YAML_PATH,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            model_name=label,
            batch=BATCH,
            patience=PATIENCE,
            cache=CACHE,
            augment=AUGMENT,
            cos_lr=COS_LR,
        )
        best_pt = os.path.join(str(results.save_dir), "weights", "best.pt")
        print(f"  ✅ {label} finished. Best weights saved to: {best_pt}")
        if metrics_path:
            print(f"  📊 Metrics saved to: {metrics_path}")
        results_summary.append((label, "✅ Success", best_pt))

    except Exception as e:
        print(f"  ❌ {label} failed with error: {e}")
        results_summary.append((label, f"❌ Failed: {e}", "—"))

print(f"\n{'='*60}")
print("  ALL TRAINING RUNS COMPLETE")
print(f"{'='*60}")
for label, status, pt in results_summary:
    print(f"  {label:12s} | {status}")
print(f"\nAll evaluation metrics are available in: {os.path.join(HERE, 'runs', 'metrics')}")
print("Launch Streamlit (`streamlit run app.py`) on this machine to visualize the Model Comparison.\n")
