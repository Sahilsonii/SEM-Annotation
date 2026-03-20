"""
balance_dataset.py
──────────────────
Fixes two imbalance problems in the SEM annotation dataset:

  Problem 1 — Class 2 (3D-2D pinholes) has only 14 images → augments to ~80
  Problem 2 — Background classes (508 images) overwhelm defect classes (158)
              → subsamples backgrounds to a configurable cap

The script writes a NEW balanced dataset folder — it never modifies your
original images or labels. You can re-run safely with different settings.

Usage:
    python balance_dataset.py
"""

import os
import shutil
import random
import math
import numpy as np
from PIL import Image, ImageEnhance
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these before running
# ──────────────────────────────────────────────────────────────────────────────

IMAGE_ROOT  = r"C:\Users\asus\Desktop\SEM annotation"
LABELS_ROOT = r"C:\Users\asus\Desktop\SEM annotation\sem_app\labels"
OUTPUT_ROOT = r"C:\Users\asus\Desktop\SEM annotation\sem_app\balanced_dataset"

# Folder name → (YOLO class ID, output class folder name)
# None = background class (empty label files, no boxes).
FOLDER_CLASS_MAP = {
    "3D perovskite with PbI2 excess":        (0, "class0_pbI2"),
    "3D perovskite with pinholes":           (1, "class1_3D_pinholes"),
    "3D-2D mixed perovskite with pinholes":  (2, "class2_3D-2D_pinholes"),  # minority — will be augmented
    "3D perovskite":                         (3, "class3_3D_background"),  # background
    "3D-2D mixed perovskite":                (4, "class4_3D-2D_background"),  # background
}

# Target image count for every defect class after augmentation.
# Class 2 has 14 → augmented to this number.
# Classes 0 and 1 already exceed this so they are just copied, not augmented.
DEFECT_TARGET = 80

# Max background images to keep per background folder.
# Keeps total background (~200) roughly equal to total defect (~240).
BG_CAP = 100

SEED = 42
IMAGE_EXTENSIONS = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"}

random.seed(SEED)
np.random.seed(SEED)


# ── Box transform helpers ─────────────────────────────────────────────────────

def flip_boxes(boxes, flip_h=False, flip_v=False):
    out = []
    for cls, xc, yc, bw, bh in boxes:
        if flip_h: xc = 1.0 - xc
        if flip_v: yc = 1.0 - yc
        out.append((cls, xc, yc, bw, bh))
    return out


def rotate_boxes_90(boxes, k):
    """Rotate boxes by k * 90 degrees counter-clockwise."""
    out = []
    for cls, xc, yc, bw, bh in boxes:
        for _ in range(k % 4):
            xc, yc = yc, 1.0 - xc
            bw, bh = bh, bw
        out.append((cls, xc, yc, bw, bh))
    return out


def augment(img: Image.Image, boxes: list, aug_id: int):
    """
    12 physically valid SEM augmentations (no elastic distortion / color jitter).
    Returns (augmented_image, augmented_boxes).
    """
    aug_id = aug_id % 12

    if aug_id == 0:   # H-flip
        img   = img.transpose(Image.FLIP_LEFT_RIGHT)
        boxes = flip_boxes(boxes, flip_h=True)
    elif aug_id == 1: # V-flip
        img   = img.transpose(Image.FLIP_TOP_BOTTOM)
        boxes = flip_boxes(boxes, flip_v=True)
    elif aug_id == 2: # HV-flip
        img   = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
        boxes = flip_boxes(boxes, flip_h=True, flip_v=True)
    elif aug_id == 3: # 90 CCW
        img   = img.rotate(90, expand=True)
        boxes = rotate_boxes_90(boxes, 1)
    elif aug_id == 4: # 180
        img   = img.rotate(180, expand=True)
        boxes = rotate_boxes_90(boxes, 2)
    elif aug_id == 5: # 270 CCW
        img   = img.rotate(270, expand=True)
        boxes = rotate_boxes_90(boxes, 3)
    elif aug_id == 6: # Brightness +10%
        img = ImageEnhance.Brightness(img).enhance(1.1)
    elif aug_id == 7: # Brightness -10%
        img = ImageEnhance.Brightness(img).enhance(0.9)
    elif aug_id == 8: # Contrast +15%
        img = ImageEnhance.Contrast(img).enhance(1.15)
    elif aug_id == 9: # H-flip + 90 CCW
        img   = img.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True)
        boxes = flip_boxes(boxes, flip_h=True)
        boxes = rotate_boxes_90(boxes, 1)
    elif aug_id == 10: # V-flip + 90 CCW
        img   = img.transpose(Image.FLIP_TOP_BOTTOM).rotate(90, expand=True)
        boxes = flip_boxes(boxes, flip_v=True)
        boxes = rotate_boxes_90(boxes, 1)
    elif aug_id == 11: # H-flip + 270 CCW
        img   = img.transpose(Image.FLIP_LEFT_RIGHT).rotate(270, expand=True)
        boxes = flip_boxes(boxes, flip_h=True)
        boxes = rotate_boxes_90(boxes, 3)

    return img, boxes


# ── Label I/O ─────────────────────────────────────────────────────────────────

def read_label(txt_path):
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) == 5:
                boxes.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return boxes


def write_label(txt_path, boxes):
    with open(txt_path, "w") as f:
        f.write("\n".join(
            f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
            for cls, xc, yc, bw, bh in boxes
        ))


# ── File helpers ──────────────────────────────────────────────────────────────

def collect_pairs(img_folder, lbl_folder):
    pairs = []
    if not os.path.exists(img_folder):
        return pairs
    for fname in sorted(os.listdir(img_folder)):
        if Path(fname).suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        pairs.append((
            os.path.join(img_folder, fname),
            os.path.join(lbl_folder, Path(fname).stem + ".txt"),
        ))
    return pairs


def save_pair(img: Image.Image, boxes, stem, out_img_dir, out_lbl_dir):
    img.convert("RGB").save(os.path.join(out_img_dir, stem + ".jpg"), "JPEG")
    write_label(os.path.join(out_lbl_dir, stem + ".txt"), boxes)


# ── Core ──────────────────────────────────────────────────────────────────────

def process_defect_class(pairs, target, out_img_dir, out_lbl_dir):
    n_orig  = len(pairs)
    needed  = max(0, target - n_orig)
    print(f"    original={n_orig}  target={target}  to_generate={needed}")

    # Copy originals.
    for img_path, txt_path in pairs:
        with Image.open(img_path) as img:
            save_pair(img.convert("RGB"), read_label(txt_path),
                      Path(img_path).stem, out_img_dir, out_lbl_dir)

    if needed == 0:
        return n_orig

    # Generate augmented copies cycling through 12 variants.
    generated = 0
    aug_round = 0
    while generated < needed:
        for i, (img_path, txt_path) in enumerate(pairs):
            if generated >= needed:
                break
            aug_id   = aug_round * n_orig + i
            new_stem = f"{Path(img_path).stem}_aug{aug_id:03d}"
            try:
                with Image.open(img_path) as img:
                    aug_img, aug_boxes = augment(img.convert("RGB"), read_label(txt_path), aug_id)
                    save_pair(aug_img, aug_boxes, new_stem, out_img_dir, out_lbl_dir)
                    generated += 1
            except Exception as e:
                print(f"    WARNING augmentation failed for {img_path}: {e}")
        aug_round += 1

    print(f"    generated {generated} augmented copies → total {n_orig + generated}")
    return n_orig + generated


def process_background(pairs, cap, out_img_dir, out_lbl_dir, tag):
    if len(pairs) > cap:
        pairs = random.sample(pairs, cap)
        print(f"    subsampled to {cap}")
    else:
        print(f"    kept all {len(pairs)} (under cap {cap})")

    for img_path, txt_path in pairs:
        stem = Path(img_path).stem + f"_{tag}"
        with Image.open(img_path) as img:
            save_pair(img.convert("RGB"), read_label(txt_path),
                      stem, out_img_dir, out_lbl_dir)
    return len(pairs)


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("=" * 65)
    print("  SEM Dataset Balancer")
    print("=" * 65)

    out_img = os.path.join(OUTPUT_ROOT, "images")
    out_lbl = os.path.join(OUTPUT_ROOT, "labels")

    if os.path.exists(OUTPUT_ROOT):
        shutil.rmtree(OUTPUT_ROOT)
    os.makedirs(out_img)
    os.makedirs(out_lbl)

    summary = {}

    for folder, class_info in FOLDER_CLASS_MAP.items():
        class_id, class_folder_name = class_info
        
        img_folder = os.path.join(IMAGE_ROOT,  folder)
        lbl_folder = os.path.join(LABELS_ROOT, folder)
        pairs      = collect_pairs(img_folder, lbl_folder)

        print(f"\n{'[Class ' + str(class_id) + ']':10s} {folder}  ({len(pairs)} images)")

        if not pairs:
            print("    SKIP — no images found")
            continue

        # Create output directories for this class
        out_class_img = os.path.join(out_img, class_folder_name)
        out_class_lbl = os.path.join(out_lbl, class_folder_name)
        os.makedirs(out_class_img, exist_ok=True)
        os.makedirs(out_class_lbl, exist_ok=True)

        # Process based on class type (defect or background)
        if class_id >= 3:  # Background class
            count = process_background(pairs, BG_CAP, out_class_img, out_class_lbl, class_folder_name)
        else:  # Defect class
            count = process_defect_class(pairs, DEFECT_TARGET, out_class_img, out_class_lbl)

        summary[class_folder_name] = (class_id, count)

    # ── Print final summary ───────────────────────────────────────────────────
    total = sum(c for _, c in summary.values())
    defect_count = sum(c for cls, c in summary.values() if cls < 3)
    bg_count = sum(c for cls, c in summary.values() if cls >= 3)
    
    print("\n" + "=" * 70)
    print("  FINAL BALANCED DATASET - CLASS BREAKDOWN")
    print("=" * 70)
    print("\n  DEFECT CLASSES:")
    for class_folder, (cls, count) in summary.items():
        if cls < 3:
            print(f"    Class {cls}: {class_folder:30s}  {count:>4} images")
    
    print("\n  BACKGROUND CLASSES:")
    for class_folder, (cls, count) in summary.items():
        if cls >= 3:
            print(f"    Class {cls}: {class_folder:30s}  {count:>4} images")
    
    print(f"\n  {'─'*70}")
    print(f"  Defect total:     {defect_count:>4} images ({100*defect_count/total:.1f}%)")
    print(f"  Background total: {bg_count:>4} images ({100*bg_count/total:.1f}%)")
    print(f"  GRAND TOTAL:      {total:>4} images")
    print(f"\n  Output folder: {OUTPUT_ROOT}")
    print(f"  Each class stored in: images/classX_*/  and  labels/classX_*/")
    print("""
  YAML Configuration for training:
    path: <path_to>/balanced_dataset
    train: images
    val: images
    test: images
    
    nc: 5
    names: ['PbI2', '3D_pinholes', '3D-2D_pinholes', '3D_background', '3D-2D_background']
  
  Next steps:
  1. Update model_handler.py to point to this balanced_dataset
  2. Train with: yolov8s train data=data.yaml epochs=200 imgsz=1024 device=cpu
  3. Monitor Class 2 (3D-2D_pinholes) recall improvement from ~30% to 60-70%
""")
    print("=" * 70)
    
    # Create data.yaml for convenience
    yaml_path = os.path.join(OUTPUT_ROOT, "data.yaml")
    yaml_content = """path: {}
train: images
val: images
test: images

nc: 5
names:
  - PbI2
  - 3D_pinholes
  - 3D-2D_pinholes
  - 3D_background
  - 3D-2D_background
""".format(OUTPUT_ROOT)
    
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"✓ Created {yaml_path}")



if __name__ == "__main__":
    run()
