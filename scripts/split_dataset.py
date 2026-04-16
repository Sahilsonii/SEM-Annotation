import os
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)

from src.utils.annotation_utils import (
    read_yolo_labels, write_yolo_labels,
    yolo_to_absolute, clip_box_to_patch, absolute_to_yolo,
    collect_image_label_pairs,
)

PATCH_SIZE = 224
OVERLAP = 0.25
STRIDE = int(PATCH_SIZE * (1 - OVERLAP))
MIN_VISIBLE_RATIO = 0.20

INPUT_DIR = os.path.join(PROJECT_ROOT, "balanced_dataset")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "balanced_dataset_split")
OUTPUT_IMG_DIR = os.path.join(OUTPUT_DIR, "images")
OUTPUT_LBL_DIR = os.path.join(OUTPUT_DIR, "labels")


def generate_patches(img, img_h, img_w):
    patches = []
    y = 0
    row = 0
    while y < img_h:
        x = 0
        col = 0
        py2 = min(y + PATCH_SIZE, img_h)
        py1 = py2 - PATCH_SIZE
        if py1 < 0:
            py1 = 0
            py2 = min(PATCH_SIZE, img_h)
        while x < img_w:
            px2 = min(x + PATCH_SIZE, img_w)
            px1 = px2 - PATCH_SIZE
            if px1 < 0:
                px1 = 0
                px2 = min(PATCH_SIZE, img_w)
            patch = img[py1:py2, px1:px2]
            patches.append((patch, px1, py1, px2, py2, row, col))
            if px2 >= img_w:
                break
            x += STRIDE
            col += 1
        if py2 >= img_h:
            break
        y += STRIDE
        row += 1
    return patches


def transform_labels_for_patch(boxes, px1, py1, px2, py2, patch_w, patch_h):
    new_boxes = []
    for cls_id, ax1, ay1, ax2, ay2 in boxes:
        result = clip_box_to_patch(cls_id, ax1, ay1, ax2, ay2, px1, py1, px2, py2, MIN_VISIBLE_RATIO)
        if result is None:
            continue
        c, cx1, cy1, cx2, cy2 = result
        local_x1 = cx1 - px1
        local_y1 = cy1 - py1
        local_x2 = cx2 - px1
        local_y2 = cy2 - py1
        local_boxes = [(c, local_x1, local_y1, local_x2, local_y2)]
        yolo_boxes = absolute_to_yolo(local_boxes, patch_w, patch_h)
        new_boxes.extend(yolo_boxes)
    return new_boxes


def process_single_image(img_path, lbl_path, class_prefix=""):
    img = cv2.imread(img_path)
    if img is None:
        return 0, 0
    img_h, img_w = img.shape[:2]
    stem = Path(img_path).stem

    yolo_boxes = read_yolo_labels(lbl_path)
    abs_boxes = yolo_to_absolute(yolo_boxes, img_w, img_h)

    patches = generate_patches(img, img_h, img_w)
    patch_count = 0
    labeled_count = 0

    for patch_img, px1, py1, px2, py2, row, col in patches:
        patch_h, patch_w = patch_img.shape[:2]
        if patch_h < 10 or patch_w < 10:
            continue

        patch_labels = transform_labels_for_patch(
            abs_boxes, px1, py1, px2, py2, patch_w, patch_h
        )

        patch_stem = f"{class_prefix}{stem}_patch_{row:02d}_{col:02d}"
        out_img_path = os.path.join(OUTPUT_IMG_DIR, patch_stem + ".jpg")
        out_lbl_path = os.path.join(OUTPUT_LBL_DIR, patch_stem + ".txt")

        cv2.imwrite(out_img_path, patch_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        write_yolo_labels(out_lbl_path, patch_labels)

        patch_count += 1
        if len(patch_labels) > 0:
            labeled_count += 1

    return patch_count, labeled_count


def run():
    print("=" * 65)
    print("  SEM Image Splitting + Label Preservation")
    print(f"  Patch Size: {PATCH_SIZE}x{PATCH_SIZE}  |  Overlap: {OVERLAP*100:.0f}%  |  Stride: {STRIDE}")
    print("=" * 65)

    if os.path.exists(OUTPUT_DIR):
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LBL_DIR, exist_ok=True)

    input_img_dir = os.path.join(INPUT_DIR, "images")
    input_lbl_dir = os.path.join(INPUT_DIR, "labels")

    all_pairs = []
    class_dirs = sorted([
        d for d in os.listdir(input_img_dir)
        if os.path.isdir(os.path.join(input_img_dir, d))
    ])

    if class_dirs:
        for cls_dir in class_dirs:
            cls_img_dir = os.path.join(input_img_dir, cls_dir)
            cls_lbl_dir = os.path.join(input_lbl_dir, cls_dir)
            pairs = collect_image_label_pairs(cls_img_dir, cls_lbl_dir)
            prefix = cls_dir + "_"
            for img_p, lbl_p in pairs:
                all_pairs.append((img_p, lbl_p, prefix))
            print(f"  Found {len(pairs)} images in {cls_dir}")
    else:
        pairs = collect_image_label_pairs(input_img_dir, input_lbl_dir)
        for img_p, lbl_p in pairs:
            all_pairs.append((img_p, lbl_p, ""))
        print(f"  Found {len(pairs)} images (flat structure)")

    total_patches = 0
    total_labeled = 0

    for img_path, lbl_path, prefix in tqdm(all_pairs, desc="Splitting"):
        pc, lc = process_single_image(img_path, lbl_path, prefix)
        total_patches += pc
        total_labeled += lc

    import yaml
    data_yaml = {
        "path": OUTPUT_DIR,
        "train": "images",
        "val": "images",
        "nc": 5,
        "names": ["PbI2", "3D_pinholes", "3D-2D_pinholes", "3D_background", "3D-2D_background"],
    }
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"\n{'=' * 65}")
    print(f"  SPLITTING COMPLETE")
    print(f"  Original images: {len(all_pairs)}")
    print(f"  Total patches:   {total_patches}")
    print(f"  Patches w/ labels: {total_labeled}")
    print(f"  Expansion ratio: {total_patches / max(len(all_pairs), 1):.1f}x")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  data.yaml: {yaml_path}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    run()
