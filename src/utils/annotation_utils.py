import os
import numpy as np
from pathlib import Path


def read_yolo_labels(txt_path):
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                boxes.append((cls_id, xc, yc, w, h))
    return boxes


def write_yolo_labels(txt_path, boxes):
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, "w") as f:
        lines = []
        for cls_id, xc, yc, w, h in boxes:
            lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        f.write("\n".join(lines))


def yolo_to_absolute(boxes, img_w, img_h):
    abs_boxes = []
    for cls_id, xc, yc, bw, bh in boxes:
        abs_xc = xc * img_w
        abs_yc = yc * img_h
        abs_w = bw * img_w
        abs_h = bh * img_h
        x1 = abs_xc - abs_w / 2.0
        y1 = abs_yc - abs_h / 2.0
        x2 = abs_xc + abs_w / 2.0
        y2 = abs_yc + abs_h / 2.0
        abs_boxes.append((cls_id, x1, y1, x2, y2))
    return abs_boxes


def absolute_to_yolo(abs_boxes, img_w, img_h):
    yolo_boxes = []
    for cls_id, x1, y1, x2, y2 in abs_boxes:
        xc = ((x1 + x2) / 2.0) / img_w
        yc = ((y1 + y2) / 2.0) / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        xc = np.clip(xc, 0.0, 1.0)
        yc = np.clip(yc, 0.0, 1.0)
        bw = np.clip(bw, 0.0, 1.0)
        bh = np.clip(bh, 0.0, 1.0)
        yolo_boxes.append((cls_id, float(xc), float(yc), float(bw), float(bh)))
    return yolo_boxes


def clip_box_to_patch(cls_id, x1, y1, x2, y2, px1, py1, px2, py2, min_visible_ratio=0.2):
    ix1 = max(x1, px1)
    iy1 = max(y1, py1)
    ix2 = min(x2, px2)
    iy2 = min(y2, py2)
    if ix1 >= ix2 or iy1 >= iy2:
        return None
    orig_area = (x2 - x1) * (y2 - y1)
    if orig_area <= 0:
        return None
    clipped_area = (ix2 - ix1) * (iy2 - iy1)
    if clipped_area / orig_area < min_visible_ratio:
        return None
    return (cls_id, ix1, iy1, ix2, iy2)


def yolo_to_coco_absolute(boxes, img_w, img_h):
    coco_boxes = []
    for cls_id, xc, yc, bw, bh in boxes:
        abs_xc = xc * img_w
        abs_yc = yc * img_h
        abs_w = bw * img_w
        abs_h = bh * img_h
        x_min = abs_xc - abs_w / 2.0
        y_min = abs_yc - abs_h / 2.0
        x_max = abs_xc + abs_w / 2.0
        y_max = abs_yc + abs_h / 2.0
        coco_boxes.append({
            "class_id": cls_id,
            "bbox": [x_min, y_min, x_max, y_max]
        })
    return coco_boxes


def collect_image_label_pairs(img_dir, lbl_dir, extensions=None):
    if extensions is None:
        extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    pairs = []
    if not os.path.isdir(img_dir):
        return pairs
    for fname in sorted(os.listdir(img_dir)):
        if Path(fname).suffix.lower() not in extensions:
            continue
        img_path = os.path.join(img_dir, fname)
        lbl_path = os.path.join(lbl_dir, Path(fname).stem + ".txt")
        pairs.append((img_path, lbl_path))
    return pairs
