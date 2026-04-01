import os
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import List, Tuple


CLAHE_CLIP        = 3.0
CLAHE_GRID        = 8
MIN_AREA_LARGE    = 40
MIN_AREA_MICRO    = 4
MAX_AREA          = 20000
MIN_CIRCULARITY   = 0.40
MIN_SOLIDITY      = 0.50
CONFIDENCE_THRESH = 0.30
NMS_IOU_THRESH    = 0.40
MAX_FILL_RATIO    = 0.06


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    area: float
    perimeter: float
    circularity: float
    solidity: float
    contrast: float
    source: str
    confidence: float


def load_image(path: str) -> Tuple[np.ndarray, np.ndarray]:
    gray  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    color = cv2.imread(path, cv2.IMREAD_COLOR)
    if gray is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return gray, color


def crop_sem_bar(gray: np.ndarray, color: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    h, w = gray.shape
    scan = gray[int(h * 0.85):, :]
    row_medians = np.median(scan, axis=1)
    dark_rows = np.where(row_medians < 15)[0]
    if len(dark_rows) > 0:
        # Cut 5 pixels above the strictly dark text background to clear the white divider line
        cut_y = int(h * 0.85) + dark_rows[0] - 5
    else:
        # Fallback to hard 12% bottom crop to safely remove metadata
        cut_y = int(h * 0.88)
    return gray[:cut_y, :], color[:cut_y, :], cut_y


def preprocess(gray: np.ndarray) -> np.ndarray:
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe    = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_GRID, CLAHE_GRID))
    return clahe.apply(denoised)


# ─── PATH A: Dark Percentile (large + medium pinholes) ─────────────────────

def dark_region_mask(enhanced: np.ndarray) -> np.ndarray:
    dark_pct = float(np.percentile(enhanced, 12))
    _, binary = cv2.threshold(enhanced, int(max(dark_pct, 3)), 255, cv2.THRESH_BINARY_INV)

    total_px = float(binary.shape[0] * binary.shape[1])
    fill = cv2.countNonZero(binary) / total_px
    if fill > MAX_FILL_RATIO:
        dark_pct = float(np.percentile(enhanced, 6))
        _, binary = cv2.threshold(enhanced, int(max(dark_pct, 3)), 255, cv2.THRESH_BINARY_INV)

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close, iterations=2)
    binary  = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k_open,  iterations=2)
    return binary


# ─── PATH B: Micro-pinhole detection (tiny sub-10px defects) ───────────────

def micro_pinhole_mask(enhanced: np.ndarray) -> np.ndarray:
    micro_thresh = float(np.percentile(enhanced, 3))
    _, binary = cv2.threshold(enhanced, int(max(micro_thresh, 3)), 255, cv2.THRESH_BINARY_INV)

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close, iterations=1)
    return binary


# ─── Feature extraction & filtering ────────────────────────────────────────

def compute_local_contrast(gray: np.ndarray, x: int, y: int, w: int, h: int,
                            contour: np.ndarray) -> float:
    pad = max(w, h, 15)
    x0  = max(0, x - pad)
    y0  = max(0, y - pad)
    x1  = min(gray.shape[1], x + w + pad)
    y1  = min(gray.shape[0], y + h + pad)
    neighborhood = gray[y0:y1, x0:x1]

    mask_full = np.zeros(gray.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask_full, [contour], -1, 255, -1)
    mask_roi = mask_full[y0:y1, x0:x1]

    interior = neighborhood[mask_roi == 255]
    exterior = neighborhood[mask_roi == 0]

    if len(interior) == 0 or len(exterior) == 0:
        return 0.0

    int_mean = float(np.mean(interior))
    ext_mean = float(np.mean(exterior))
    if ext_mean < 1.0:
        return 0.0
    return float(np.clip((ext_mean - int_mean) / ext_mean, 0.0, 1.0))


def compute_confidence(area: float, circularity: float, solidity: float,
                        contrast: float, is_micro: bool) -> float:
    area_norm  = float(np.clip(area / 2000.0, 0.0, 1.0))
    micro_bonus = 0.08 if is_micro else 0.0
    conf = (0.40 * contrast +
            0.25 * circularity +
            0.15 * solidity +
            0.10 * area_norm +
            0.10 +
            micro_bonus)
    return float(np.clip(conf, 0.0, 1.0))


def extract_detections(mask: np.ndarray, gray_orig: np.ndarray,
                        source: str,
                        min_area: int = MIN_AREA_LARGE,
                        max_area: int = MAX_AREA,
                        min_circ: float = MIN_CIRCULARITY,
                        min_solid: float = MIN_SOLIDITY,
                        conf_thresh: float = CONFIDENCE_THRESH) -> List[Detection]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets         = []
    img_median   = float(np.median(gray_orig))
    img_std      = float(np.std(gray_orig))
    dark_gate    = img_median - 0.75 * img_std

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        perim = cv2.arcLength(c, True)
        if perim < 1.0:
            continue

        circ = float(4 * np.pi * area / (perim ** 2))
        
        # Relax circularity for large pinholes (they often have complex irregular shapes)
        dynamic_min_circ = min_circ
        if area > 500:
            dynamic_min_circ = max(0.15, min_circ - 0.25)
        elif area > 200:
            dynamic_min_circ = max(0.25, min_circ - 0.15)
            
        if circ < dynamic_min_circ:
            continue

        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = float(area / (hull_area + 1e-6))
        if solidity < min_solid:
            continue

        x, y, w, h = cv2.boundingRect(c)
        aspect = float(max(w, h)) / float(min(w, h) + 1e-6)
        if aspect > 4.0:
            continue

        mask_tmp = np.zeros(gray_orig.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask_tmp, [c], -1, 255, -1)
        interior_mean = float(np.mean(gray_orig[mask_tmp == 255]))
        if interior_mean > dark_gate:
            continue

        contrast = compute_local_contrast(gray_orig, int(x), int(y), int(w), int(h), c)
        if contrast < 0.15:
            continue

        is_micro = (source == "micro")
        conf = compute_confidence(float(area), circ, solidity, contrast, is_micro)

        if conf >= conf_thresh:
            dets.append(Detection(
                x=int(x), y=int(y), w=int(w), h=int(h),
                area=float(area), perimeter=float(perim),
                circularity=float(circ), solidity=float(solidity),
                contrast=float(contrast), source=source,
                confidence=float(conf),
            ))

    return dets


# ─── NMS ────────────────────────────────────────────────────────────────────

def iou(a, b):
    xa = max(a[0], b[0])
    ya = max(a[1], b[1])
    xb = min(a[0]+a[2], b[0]+b[2])
    yb = min(a[1]+a[3], b[1]+b[3])
    inter = max(0, xb-xa) * max(0, yb-ya)
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter / (union + 1e-6)


def nms(detections: List[Detection], iou_thresh: float = NMS_IOU_THRESH) -> List[Detection]:
    if not detections:
        return []
    dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    keep = []
    for d in dets:
        box = (d.x, d.y, d.w, d.h)
        if all(iou(box, (k.x, k.y, k.w, k.h)) < iou_thresh for k in keep):
            keep.append(d)
    return keep


# ─── Visualization & saving ────────────────────────────────────────────────

def annotate_image(color: np.ndarray, detections: List[Detection]) -> np.ndarray:
    out = color.copy()
    for d in detections:
        if d.source == "micro":
            c = (255, 100, 255)
            t = 1
        elif d.confidence > 0.55:
            c = (0, 255, 80)
            t = 2
        else:
            c = (0, 200, 255)
            t = 2
        cv2.rectangle(out, (d.x, d.y), (d.x + d.w, d.y + d.h), c, t)
        label = f"{d.confidence:.2f}"
        cv2.putText(out, label, (d.x, max(d.y - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1, cv2.LINE_AA)
    return out


def save_yolo(detections: List[Detection], img_shape: Tuple[int, int],
              path: str, class_id: int = 1):
    h_img, w_img = img_shape
    lines = []
    for d in detections:
        xc = (d.x + d.w / 2) / w_img
        yc = (d.y + d.h / 2) / h_img
        wn = d.w / w_img
        hn = d.h / h_img
        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))


def save_json(detections: List[Detection], image_path: str, path: str):
    payload = {
        "image": os.path.basename(image_path),
        "num_detections": len(detections),
        "detections": [asdict(d) for d in detections]
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def visualize_pipeline(stages: dict, output_path: str):
    n = len(stages)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (title, img) in zip(axes, stages.items()):
        if len(img.shape) == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── Main pipeline ─────────────────────────────────────────────────────────

def run_single(image_path: str, output_dir: str,
               conf_thresh: float = CONFIDENCE_THRESH) -> List[Detection]:
    gray_raw, color_raw = load_image(image_path)
    base = os.path.splitext(os.path.basename(image_path))[0]

    gray, color, cut_y = crop_sem_bar(gray_raw, color_raw)
    enhanced = preprocess(gray)

    mask_large = dark_region_mask(enhanced)
    dets_large = extract_detections(mask_large, gray, "large",
                                     min_area=MIN_AREA_LARGE,
                                     conf_thresh=conf_thresh)

    mask_micro = micro_pinhole_mask(enhanced)
    dets_micro = extract_detections(mask_micro, gray, "micro",
                                     min_area=MIN_AREA_MICRO,
                                     min_circ=0.50,
                                     min_solid=0.55,
                                     conf_thresh=conf_thresh)

    all_dets   = dets_large + dets_micro
    detections = nms(all_dets)

    annotated = annotate_image(color, detections)

    stages = {
        "CLAHE": enhanced,
        "Dark Mask (large)": mask_large,
        "Micro Mask (tiny)": mask_micro,
        f"Detections ({len(detections)})": annotated,
    }

    vis_path  = os.path.join(output_dir, "visualizations", f"{base}_pipeline.png")
    yolo_path = os.path.join(output_dir, "annotations", f"{base}.txt")
    json_path = os.path.join(output_dir, "annotations", f"{base}.json")
    ann_path  = os.path.join(output_dir, "visualizations", f"{base}_result.jpg")

    visualize_pipeline(stages, vis_path)
    save_yolo(detections, gray.shape, yolo_path)
    save_json(detections, image_path, json_path)
    cv2.imwrite(ann_path, annotated)

    return detections


def run_folder(input_dir: str, output_dir: str,
               extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff")) -> dict:
    results = {}
    import os
    for root, _, files in os.walk(input_dir):
        for fname in sorted(files):
            if not fname.lower().endswith(extensions):
                continue
            if "aug" in fname.lower():
                print(f"  Skipping augmented image: {fname}")
                continue
            fpath = os.path.join(root, fname)
            dets  = run_single(fpath, output_dir)
            results[fname] = len(dets)
            print(f"  {fname}: {len(dets)} detections")
    return results
