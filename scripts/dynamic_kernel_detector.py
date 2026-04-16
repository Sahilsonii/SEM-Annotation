import os
import sys
import cv2
import json
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)

from src.filters.adaptive_kernel import AdaptiveKernelEngine
from src.filters.classical_blend import ClassicalFilterBlend
from src.utils.annotation_utils import write_yolo_labels, collect_image_label_pairs

EXP_DIR = os.path.join(PROJECT_ROOT, "experiments", "dynamic_kernel")
FILTERED_DIR = os.path.join(EXP_DIR, "filtered_images")
MASKS_DIR = os.path.join(EXP_DIR, "masks")
BBOX_DIR = os.path.join(EXP_DIR, "bbox_overlays")
LABELS_DIR = os.path.join(EXP_DIR, "labels")

INPUT_DIR = os.path.join(PROJECT_ROOT, "balanced_dataset", "images")

CLAHE_CLIP = 3.0
CLAHE_GRID = 8
GAUSSIAN_BLUR_SIGMA = 1.0
MIN_DEFECT_AREA = 30
MAX_DEFECT_AREA = 50000


def setup_directories():
    for d in [FILTERED_DIR, MASKS_DIR, BBOX_DIR, LABELS_DIR]:
        os.makedirs(d, exist_ok=True)


def preprocess(gray):
    clahe = ClassicalFilterBlend.apply_clahe(gray, CLAHE_CLIP, CLAHE_GRID)
    ksize = int(np.ceil(GAUSSIAN_BLUR_SIGMA * 6)) | 1
    smoothed = cv2.GaussianBlur(clahe, (ksize, ksize), GAUSSIAN_BLUR_SIGMA)
    return smoothed


def compute_enhanced_response(gray, engine, blend):
    variance_map = engine.compute_local_variance_map(gray, window=21)
    gradient_map = engine.compute_gradient_magnitude_map(gray)
    entropy_map = engine.compute_fast_entropy_map(gray, window=15)

    kernel_size_map, complexity_map = engine.compute_kernel_size_map(
        variance_map, gradient_map, entropy_map
    )

    response, intermediates = engine.compute_hybrid_response(
        gray, kernel_size_map, complexity_map
    )

    gabor = blend.gabor_filter_bank(gray, num_orientations=6, frequencies=[0.05, 0.1, 0.15])
    hessian = blend.hessian_blob_detector(gray, sigmas=[1.0, 2.0, 4.0])
    lbp_var = blend.local_binary_pattern_variance(gray, radius=3)
    dct_hf = blend.dct_highfreq_response(gray, block_size=32)
    median_dev = blend.median_deviation(gray, kernel_size=7)
    unsharp = blend.unsharp_mask(gray, sigma=3.0, strength=1.5)

    def _norm(m):
        mn, mx = m.min(), m.max()
        if mx - mn < 1e-8:
            return np.zeros_like(m, dtype=np.float64)
        return (m - mn) / (mx - mn)

    gabor_n = _norm(gabor)
    hessian_n = _norm(hessian)
    lbp_n = _norm(lbp_var)
    dct_n = _norm(dct_hf)
    median_n = _norm(median_dev)
    response_n = _norm(response)

    enhanced = (
        0.35 * response_n +
        0.15 * gabor_n +
        0.15 * hessian_n +
        0.10 * lbp_n +
        0.10 * dct_n +
        0.15 * median_n
    )

    enhanced = (enhanced * 255).astype(np.uint8)

    return enhanced, kernel_size_map, intermediates


def process_image(img_path, engine, blend, save_intermediates=True):
    img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_color is None:
        return None, {}
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    stem = Path(img_path).stem

    preprocessed = preprocess(gray)

    enhanced, kernel_size_map, intermediates = compute_enhanced_response(
        preprocessed, engine, blend
    )

    binary = engine.hybrid_threshold(enhanced.astype(np.float64))

    detections = engine.extract_defects(binary, gray, MIN_DEFECT_AREA, MAX_DEFECT_AREA)

    if save_intermediates:
        cv2.imwrite(os.path.join(FILTERED_DIR, f"{stem}_enhanced.jpg"), enhanced)
        cv2.imwrite(os.path.join(MASKS_DIR, f"{stem}_mask.png"), binary)

        overlay = img_color.copy()
        colors = {0: (0, 255, 255), 1: (0, 255, 0), 2: (255, 0, 0)}
        type_labels = {0: "PbI2", 1: "sm_pin", 2: "lg_pin"}
        for det in detections:
            x, y, w, h = det["bbox_abs"]
            cid = det["class_id"]
            conf = det["confidence"]
            color = colors.get(cid, (255, 255, 255))
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            label = f"{type_labels.get(cid, '?')} {conf:.2f}"
            cv2.putText(overlay, label, (x, max(y - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(BBOX_DIR, f"{stem}_overlay.jpg"), overlay)

    img_h, img_w = gray.shape
    yolo_boxes = []
    for det in detections:
        xc, yc, bw, bh = det["bbox_yolo"]
        yolo_boxes.append((det["class_id"], xc, yc, bw, bh))

    lbl_path = os.path.join(LABELS_DIR, f"{stem}.txt")
    write_yolo_labels(lbl_path, yolo_boxes)

    stats = engine.get_kernel_stats()
    stats["num_detections"] = len(detections)
    stats["defect_types"] = {}
    for det in detections:
        dt = det["defect_type"]
        stats["defect_types"][dt] = stats["defect_types"].get(dt, 0) + 1

    return detections, stats


def run(input_dir=None, save_intermediates=True, max_images=None):
    setup_directories()
    if input_dir is None:
        input_dir = INPUT_DIR

    engine = AdaptiveKernelEngine(
        variance_weight=0.40,
        gradient_weight=0.35,
        entropy_weight=0.25,
    )
    blend = ClassicalFilterBlend()

    image_files = []
    if os.path.isdir(input_dir):
        for root, _, files in os.walk(input_dir):
            for fname in sorted(files):
                ext = os.path.splitext(fname)[1].lower()
                if ext in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
                    image_files.append(os.path.join(root, fname))

    if max_images:
        image_files = image_files[:max_images]

    print("=" * 65)
    print("  Dynamic Adaptive Kernel (DAK) Defect Detector")
    print(f"  Images: {len(image_files)}")
    print(f"  Kernel sizes: {list(AdaptiveKernelEngine.KERNEL_SIZES)}")
    print("=" * 65)

    all_stats = {}
    all_kernel_sizes = {}
    total_detections = 0
    timing = []

    for img_path in tqdm(image_files, desc="DAK Processing"):
        stem = Path(img_path).stem
        t0 = time.time()
        detections, stats = process_image(img_path, engine, blend, save_intermediates)
        elapsed = time.time() - t0
        timing.append(elapsed)

        if detections is not None:
            total_detections += len(detections)
            all_stats[stem] = stats
            if "kernel_distribution" in stats:
                all_kernel_sizes[stem] = stats["kernel_distribution"]

    kernel_json_path = os.path.join(EXP_DIR, "kernel_sizes_used.json")
    with open(kernel_json_path, "w") as f:
        json.dump(all_kernel_sizes, f, indent=2)

    perf_metrics = {
        "total_images": len(image_files),
        "total_detections": total_detections,
        "avg_detections_per_image": total_detections / max(len(image_files), 1),
        "avg_time_per_image_s": float(np.mean(timing)) if timing else 0,
        "total_time_s": float(np.sum(timing)),
        "min_time_s": float(np.min(timing)) if timing else 0,
        "max_time_s": float(np.max(timing)) if timing else 0,
        "images_per_second": len(image_files) / max(np.sum(timing), 1e-6),
    }

    defect_type_totals = {}
    for stem, stats in all_stats.items():
        for dt, count in stats.get("defect_types", {}).items():
            defect_type_totals[dt] = defect_type_totals.get(dt, 0) + count
    perf_metrics["defect_type_totals"] = defect_type_totals

    perf_json_path = os.path.join(EXP_DIR, "performance_metrics.json")
    with open(perf_json_path, "w") as f:
        json.dump(perf_metrics, f, indent=2)

    print(f"\n{'=' * 65}")
    print(f"  DAK DETECTION COMPLETE")
    print(f"  Total images processed: {len(image_files)}")
    print(f"  Total detections: {total_detections}")
    print(f"  Avg time/image: {perf_metrics['avg_time_per_image_s']:.3f}s")
    print(f"  Throughput: {perf_metrics['images_per_second']:.1f} img/s")
    print(f"  Defect breakdown: {defect_type_totals}")
    print(f"  Outputs: {EXP_DIR}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--no-intermediates", action="store_true")
    args = parser.parse_args()
    run(
        input_dir=args.input,
        save_intermediates=not args.no_intermediates,
        max_images=args.max_images,
    )
