import os
import sys
import json
import time
import logging
from datetime import datetime

import random
from collections import Counter
import shutil

# ── LOGGING SETUP ─────────────────────────────────────────────────────────────

HERE         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)

LOG_DIR = os.path.join(PROJECT_ROOT, "experiments", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")


class _TeeLogger:
    """Writes every print() call to both console and the log file."""
    def __init__(self, filepath, stream=None):
        self._file   = open(filepath, "a", encoding="utf-8", buffering=1)
        self._stream = stream or sys.__stdout__
    def write(self, msg):
        self._stream.write(msg)
        self._file.write(msg)
    def flush(self):
        self._stream.flush()
        self._file.flush()
    def close(self):
        self._file.close()

sys.stdout = _TeeLogger(LOG_FILE, sys.__stdout__)

_file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
_file_handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s: %(message)s"))
_console_handler = logging.StreamHandler(sys.__stdout__)
_console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

logging.basicConfig(level=logging.INFO, handlers=[_file_handler, _console_handler])
logger = logging.getLogger(__name__)

print(f"\n  [LOG] Log file: {LOG_FILE}\n")


# ── CONFIG ────────────────────────────────────────────────────────────────────
EPOCHS      = 50
IMGSZ       = 224
BATCH       = 4
WORKERS     = 0
SAVE_PERIOD = 10

PATIENCE = 15
CACHE    = False
AUGMENT  = True
COS_LR   = True

MODELS_TO_TRAIN = [
    {"label": "yolo11s",  "weights": "yolo11s.pt"},
    {"label": "yolo11m",  "weights": "yolo11m.pt"},
    {"label": "yolo11l",  "weights": "yolo11l.pt"},
]


# 80/10/10 Split Config
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

# ─────────────────────────────────────────────────────────────────────────────
# NEW: Source dataset has class-named subfolders, split dataset uses txt-index

SOURCE_DIR = os.path.join(PROJECT_ROOT, "dataset_split_224")         # nested per-class folders
SPLIT_DIR  = SOURCE_DIR                                             # where splits live
YAML_PATH  = os.path.join(SPLIT_DIR, "data.yaml")

# Known problematic samples that can block image decoding during validation.
KNOWN_BAD_IMAGES = {
    os.path.join(SPLIT_DIR, "images", "5-5_patch_4_5.jpg"),
}

# Class folder → (class_id, display_name)
CLASS_MAP = {
    "class0_pbI2":             ("0", "PbI2           [DEFECT]"),
    "class1_3D_pinholes":      ("1", "3D Pinholes    [DEFECT]"),
    "class2_3D-2D_pinholes":   ("2", "3D-2D Pinholes [DEFECT]"),
    "class3_3D_background":    ("3", "3D Background"),
    "class4_3D-2D_background": ("4", "3D-2D Background"),
}

sys.path.insert(0, PROJECT_ROOT)


def organize_stratified_split(source_root, split_root, train_ratio=0.8, val_ratio=0.1):
    """
    Creates stratified train.txt / val.txt / test.txt index files by scanning
    the per-class subfolders inside `balanced_dataset/images/classN_*/`.
    NO FILES ARE MOVED. Fully resumable.
    """
    src_img_root = os.path.join(source_root, "images")
    src_lbl_root = os.path.join(source_root, "labels")

    os.makedirs(split_root, exist_ok=True)
    train_txt = os.path.join(split_root, "train.txt")
    val_txt   = os.path.join(split_root, "val.txt")
    test_txt  = os.path.join(split_root, "test.txt")

    def _sanitize_split_files(txt_paths):
        removed_total = 0
        for txt_path in txt_paths:
            if not os.path.exists(txt_path):
                continue
            with open(txt_path, "r") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            clean = [ln for ln in lines if os.path.normcase(os.path.abspath(ln)) not in
                     {os.path.normcase(os.path.abspath(p)) for p in KNOWN_BAD_IMAGES}]
            removed = len(lines) - len(clean)
            if removed > 0:
                with open(txt_path, "w") as f:
                    f.write("\n".join(clean))
                removed_total += removed
                print(f"  Removed {removed} known bad sample(s) from {os.path.basename(txt_path)}")
        return removed_total

    # ── Already split? Print counts and return ────────────────────────────────
    if all(os.path.exists(p) for p in (train_txt, val_txt, test_txt)):
        _sanitize_split_files((train_txt, val_txt, test_txt))
        n_tr = sum(1 for _ in open(train_txt))
        n_va = sum(1 for _ in open(val_txt))
        n_te = sum(1 for _ in open(test_txt))
        if n_tr > 0 and n_va > 0 and n_te > 0:
            print(f"\n  Split index files exist  ->  Train: {n_tr:,} | Val: {n_va:,} | Test: {n_te:,}")
            return

    if not os.path.isdir(src_img_root):
        raise FileNotFoundError(f"Source images/ not found in {source_root}")
    if not os.path.isdir(src_lbl_root):
        raise FileNotFoundError(f"Source labels/ not found in {source_root}")

    # ── Enumerate images by class subfolder ───────────────────────────────────
    groups = {cls_id: [] for _, (cls_id, _) in CLASS_MAP.items()}
    missing_classes = []

    for folder_name, (cls_id, _) in CLASS_MAP.items():
        cls_img_dir = os.path.join(src_img_root, folder_name)
        if not os.path.isdir(cls_img_dir):
            missing_classes.append(folder_name)
            continue
        for fname in sorted(os.listdir(cls_img_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.abspath(os.path.join(cls_img_dir, fname))
                if os.path.normcase(img_path) in {os.path.normcase(os.path.abspath(p)) for p in KNOWN_BAD_IMAGES}:
                    continue
                groups[cls_id].append(img_path)

    if missing_classes:
        print(f"  ⚠ Missing class folders (skipped): {missing_classes}")

    total = sum(len(v) for v in groups.values())
    if total == 0:
        raise RuntimeError(f"No images found under {src_img_root}")

    print(f"\n{'='*60}")
    print(f"  STRATIFIED SPLIT  (80% Train / 10% Val / 10% Test)")
    print(f"  Source: {source_root}  (nested class folders)")
    print(f"  Total patches: {total:,}  |  Method: index files")
    print(f"{'='*60}")

    print(f"\n  {'Class':<6} {'Name':<28} {'Patches':>8}  {'Type'}")
    print(f"  {'-'*55}")
    for prefix, (cls_id, name) in CLASS_MAP.items():
        count = len(groups[cls_id])
        tag = "DEFECT" if cls_id in ("0", "1", "2") else "background"
        print(f"  {cls_id:<6} {name:<28} {count:>8,}  {tag}")
    print(f"  {'TOTAL':<35} {total:>8,}")

    # ── Build stratified split lists ──────────────────────────────────────────
    random.seed(42)
    split_lists = {"train": [], "val": [], "test": []}
    split_stats = {"train": Counter(), "val": Counter(), "test": Counter()}

    for cls_id, paths in groups.items():
        if not paths:
            continue
        random.shuffle(paths)
        n_train = int(len(paths) * train_ratio)
        n_val   = int(len(paths) * val_ratio)
        split_lists["train"].extend(paths[:n_train])
        split_lists["val"].extend(paths[n_train:n_train + n_val])
        split_lists["test"].extend(paths[n_train + n_val:])
        split_stats["train"][cls_id] += n_train
        split_stats["val"][cls_id]   += n_val
        split_stats["test"][cls_id]  += len(paths) - n_train - n_val

    # ── Shuffle each split (so classes are mixed) ─────────────────────────────
    for k in split_lists:
        random.shuffle(split_lists[k])

    # ── Write txt files ───────────────────────────────────────────────────────
    for split_name, txt_path in [("train", train_txt), ("val", val_txt), ("test", test_txt)]:
        with open(txt_path, "w") as f:
            f.write("\n".join(split_lists[split_name]))
        print(f"  Written: {os.path.basename(txt_path)}  ({len(split_lists[split_name]):,} entries)")

    _sanitize_split_files((train_txt, val_txt, test_txt))

    # ── Final report ──────────────────────────────────────────────────────────
    cls_display = {cls_id: name for _, (cls_id, name) in CLASS_MAP.items()}

    print(f"\n{'='*60}")
    print("  SPLIT COMPLETE — FINAL CLASS DISTRIBUTION")
    print(f"{'='*60}")
    print(f"  {'Class':<6} {'Name':<28} {'Train':>7} {'Val':>7} {'Test':>7}")
    print(f"  {'-'*57}")
    for cls_id in sorted(cls_display.keys()):
        tr = split_stats["train"].get(cls_id, 0)
        va = split_stats["val"].get(cls_id, 0)
        te = split_stats["test"].get(cls_id, 0)
        if tr + va + te == 0:
            continue
        print(f"  {cls_id:<6} {cls_display[cls_id]:<28} {tr:>7,} {va:>7,} {te:>7,}")
    print(f"  {'-'*57}")
    print(f"  {'TOTAL':<34} "
          f"{len(split_lists['train']):>7,} "
          f"{len(split_lists['val']):>7,} "
          f"{len(split_lists['test']):>7,}")
    print(f"{'='*60}\n")


# ── YOLO TRAINING ─────────────────────────────────────────────────────────────

def train_yolo_models():
    if not os.path.exists(SOURCE_DIR):
        logger.error(f"Source dataset not found at {SOURCE_DIR}")
        return []

    # Build stratified split index files
    organize_stratified_split(SOURCE_DIR, SPLIT_DIR, TRAIN_RATIO, VAL_RATIO)

    import yaml as _yaml
    classes = {0: "PbI2", 1: "3D_pinholes", 2: "3D-2D_pinholes",
               3: "3D_background", 4: "3D-2D_background"}
    names   = [classes[k] for k in sorted(classes)]

    # IMPORTANT: YOLO maps images/ → labels/ by swapping the folder segment.
    # Since our txt files contain absolute paths to `balanced_dataset/images/classN_*/foo.jpg`,
    # YOLO will automatically look for labels at `balanced_dataset/labels/classN_*/foo.txt`.
    # No extra config needed.

    yaml_data = {
        "path": SPLIT_DIR,
        "train": os.path.join(SPLIT_DIR, "train.txt"),
        "val":   os.path.join(SPLIT_DIR, "val.txt"),
        "test":  os.path.join(SPLIT_DIR, "test.txt"),
        "nc": len(classes),
        "names": names
    }
    with open(YAML_PATH, "w") as f:
        _yaml.dump(yaml_data, f, default_flow_style=False)
    print(f"[OK] data.yaml (Stratified txt-index) validated at: {YAML_PATH}")

    try:
        from model_handler import ModelHandler
    except ImportError:
        try:
            from src.model_handler import ModelHandler
        except ImportError:
            logger.error("Could not import ModelHandler.")
            return []

    # ── Device Detection ──────────────────────────────────────────────────────
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        device_str = f"GPU  --  {gpu_name}  ({vram_gb:.1f} GB VRAM)"
        device_icon = "GPU"
    else:
        import platform
        device_str = f"CPU  --  {platform.processor()}"
        device_icon = "CPU"

    # ── Architecture Summary Table ────────────────────────────────────────────
    ARCH_TABLE = [
        ("yolov8s",    "CSPDarknet-S",   "C2f+PAN", "Detect",  11.2,   28.7,  "YOLO"),
        ("yolov8m",    "CSPDarknet-M",   "C2f+PAN", "Detect",  25.9,   79.0,  "YOLO"),
        ("yolov8l",    "CSPDarknet-L",   "C2f+PAN", "Detect",  43.7,  165.2,  "YOLO"),
    ]

    print(f"\n{'='*72}")
    print(f"  TRAINING SESSION — {len(MODELS_TO_TRAIN)} YOLO MODELS  |  {device_icon}: {device_str}")
    print(f"{'='*72}")
    print(f"  {'#':<3} {'Model':<12} {'Backbone':<20} {'Neck':<10} {'Head':<10} {'Params(M)':>9} {'GFLOPs':>7}  Type")
    print(f"  {'-'*70}")
    for i, (lbl, bb, neck, head, pm, gf, mtype) in enumerate(ARCH_TABLE, 1):
        print(f"  {i:<3} {lbl:<12} {bb:<20} {neck:<10} {head:<10} {pm:>9.1f} {str(gf):>7}  {mtype}")
    print(f"  {'-'*70}")
    print(f"  Total trainable params: ~{sum(r[4] for r in ARCH_TABLE):.0f}M across all models")
    print(f"\n  Training Config:")
    print(f"    Image size  : {IMGSZ}x{IMGSZ}")
    print(f"    Batch size  : {BATCH} (auto-adjusted per model size)")
    print(f"    Epochs      : {EPOCHS}  (checkpointing every {SAVE_PERIOD} epochs)")
    print(f"    Workers     : {WORKERS}  (prevents Windows freezing)")
    print(f"    Classes     : 5  (3 defects + 2 backgrounds)")
    print(f"    Source      : {SOURCE_DIR}")
    print(f"    Split dir   : {SPLIT_DIR}")
    print(f"    Device      : {device_str}")
    print(f"{'='*72}\n")

    results_summary = []

    VRAM_SAFE_BATCH = {"s": 16, "m": 8, "l": 4, "x": 2}

    import gc
    import torch

    for i, m in enumerate(MODELS_TO_TRAIN, 1):
        label   = m["label"]
        weights = m["weights"]
        wpath   = os.path.join(HERE, weights)

        # Check if model already trained
        runs_dir = os.path.join(PROJECT_ROOT, "src", "runs", "detect")
        existing_runs = [d for d in os.listdir(runs_dir) if d.startswith(label) and os.path.isdir(os.path.join(runs_dir, d))]
        
        if existing_runs:
            # Check if any run has weights
            has_weights = False
            for run_dir in existing_runs:
                weights_path = os.path.join(runs_dir, run_dir, "weights", "best.pt")
                if os.path.exists(weights_path):
                    has_weights = True
                    print(f"\n[{i}/{len(MODELS_TO_TRAIN)}] ✓ {label} already trained")
                    print(f"  Weights found: {weights_path}")
                    results_summary.append((label, "Already trained", weights_path))
                    break
            
            if has_weights:
                continue

        if not os.path.exists(wpath) and not (weights.startswith("yolov8") or weights.startswith("yolo11")):
            print(f"  ⚠️ Skipping {label}: weights {weights} not found")
            continue

        size_key = label[-1] if label[-1] in VRAM_SAFE_BATCH else "s"
        batch = VRAM_SAFE_BATCH.get(size_key, BATCH)

        print(f"\n[{i}/{len(MODELS_TO_TRAIN)}] ▶ Training {label}  (weights: {weights}, batch: {batch})")
        print("-" * 60)

        try:
            # Pass the weights name directly - Ultralytics will download if needed
            handler = ModelHandler(weights)
            results, metrics_path = handler.train_model(
                YAML_PATH,
                epochs=EPOCHS,
                imgsz=IMGSZ,
                model_name=label,
                batch=batch,
                workers=WORKERS,
                save_period=SAVE_PERIOD,
                patience=PATIENCE,
                cache=CACHE,
                augment=AUGMENT,
                cos_lr=COS_LR,
            )
            best_pt = os.path.join(str(results.save_dir), "weights", "best.pt")
            print(f"  [SUCCESS] {label} finished. Best weights saved to: {best_pt}")
            if metrics_path:
                print(f"  [DATA] Metrics saved to: {metrics_path}")
            results_summary.append((label, "Success", best_pt))

        except Exception as e:
            print(f"  [ERROR] {label} failed with error: {e}")
            results_summary.append((label, f"Failed: {e}", "—"))

        if 'handler' in locals():
            del handler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results_summary





# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train YOLO models")
    args = parser.parse_args()

    results_summary = []
    
    yolo_results = train_yolo_models()
    results_summary.extend(yolo_results)

    print(f"\n{'='*60}")
    print("  ALL TRAINING RUNS COMPLETE")
    print(f"{'='*60}")
    for label, status, pt in results_summary:
        print(f"  {label:16s} | {status}")
    print()
    
    # Generate benchmark PDF
    print(f"\n{'='*60}")
    print("  GENERATING BENCHMARK PDF REPORT")
    print(f"{'='*60}")
    try:
        import subprocess
        pdf_script = os.path.join(HERE, "create_benchmark_pdf.py")
        result = subprocess.run([sys.executable, pdf_script], cwd=PROJECT_ROOT)
        if result.returncode == 0:
            print("\n[SUCCESS] Benchmark PDF generated!")
    except Exception as e:
        print(f"\n[WARNING] Could not generate PDF: {e}")
        print("  Run manually: python scripts/create_benchmark_pdf.py")