import os
import time
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from amcaa_pipeline import (
    load_image, crop_sem_bar, preprocess,
    dark_region_mask, micro_pinhole_mask,
    extract_detections, nms,
    CONFIDENCE_THRESH, MIN_AREA_LARGE, MIN_AREA_MICRO
)


ABLATION_CONFIGS = [
    {
        "name":            "Full AMCAA (Large+Micro)",
        "use_large":       True,
        "use_micro":       True,
        "use_confidence":  True,
    },
    {
        "name":            "Large Pinholes Only",
        "use_large":       True,
        "use_micro":       False,
        "use_confidence":  True,
    },
    {
        "name":            "Micro Pinholes Only",
        "use_large":       False,
        "use_micro":       True,
        "use_confidence":  True,
    },
    {
        "name":            "No Confidence Filter",
        "use_large":       True,
        "use_micro":       True,
        "use_confidence":  False,
    },
]


def run_config_on_image(image_path, cfg):
    gray_raw, color_raw = load_image(image_path)
    gray, color, _ = crop_sem_bar(gray_raw, color_raw)
    t0 = time.perf_counter()
    enhanced = preprocess(gray)
    all_dets = []
    conf_t = CONFIDENCE_THRESH if cfg["use_confidence"] else 0.0

    if cfg["use_large"]:
        mask = dark_region_mask(enhanced)
        all_dets += extract_detections(mask, gray, "large",
                                        min_area=MIN_AREA_LARGE, conf_thresh=conf_t)
    if cfg["use_micro"]:
        mask = micro_pinhole_mask(enhanced)
        all_dets += extract_detections(mask, gray, "micro",
                                        min_area=MIN_AREA_MICRO,
                                        min_circ=0.50, min_solid=0.55,
                                        conf_thresh=conf_t)

    detections = nms(all_dets)
    elapsed = time.perf_counter() - t0
    return len(detections), elapsed


def run_ablation(image_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    results = {cfg["name"]: {"detections": [], "times": []} for cfg in ABLATION_CONFIGS}

    for img_path in image_paths:
        for cfg in ABLATION_CONFIGS:
            n_det, t = run_config_on_image(img_path, cfg)
            results[cfg["name"]]["detections"].append(n_det)
            results[cfg["name"]]["times"].append(t)

    summary = {}
    for name, data in results.items():
        summary[name] = {
            "mean_detections": float(np.mean(data["detections"])),
            "std_detections":  float(np.std(data["detections"])),
            "mean_time_s":     float(np.mean(data["times"])),
        }

    _plot_ablation(summary, output_dir)
    return summary


def _plot_ablation(summary, output_dir):
    names = list(summary.keys())
    means = [summary[n]["mean_detections"] for n in names]
    stds  = [summary[n]["std_detections"]  for n in names]
    times = [summary[n]["mean_time_s"]     for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#2196F3" if "Full" in n else "#90CAF9" for n in names]

    bars = ax1.barh(names, means, xerr=stds, color=colors, edgecolor="black", linewidth=0.7)
    ax1.set_xlabel("Mean Detections per Image")
    ax1.set_title("Ablation: Detection Count")
    ax1.bar_label(bars, fmt="%.1f", padding=3)

    t_bars = ax2.barh(names, times, color=colors, edgecolor="black", linewidth=0.7)
    ax2.set_xlabel("Mean Inference Time (s)")
    ax2.set_title("Ablation: Inference Speed")
    ax2.bar_label(t_bars, fmt="%.3f", padding=3)

    plt.tight_layout()
    path = os.path.join(output_dir, "ablation_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Ablation plot saved: {path}")

    with open(os.path.join(output_dir, "ablation_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
