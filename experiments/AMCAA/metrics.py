import numpy as np
from typing import List, Tuple


def compute_iou(box1: Tuple, box2: Tuple) -> float:
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / (union + 1e-6)


def match_detections(pred_boxes: List[Tuple], gt_boxes: List[Tuple],
                     iou_thresh: float = 0.5) -> Tuple[int, int, int]:
    tp = fp = fn = 0
    matched = set()
    for pb in pred_boxes:
        best_iou, best_j = 0.0, -1
        for j, gb in enumerate(gt_boxes):
            if j in matched:
                continue
            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh and best_j >= 0:
            tp += 1
            matched.add(best_j)
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched)
    return tp, fp, fn


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = tp / (tp + fp + 1e-6)
    rec  = tp / (tp + fn + 1e-6)
    f1   = 2 * prec * rec / (prec + rec + 1e-6)
    return prec, rec, f1


def compute_map(pred_list: List[List[Tuple]], gt_list: List[List[Tuple]],
                iou_thresh: float = 0.5) -> float:
    all_tp = all_fp = all_fn = 0
    for preds, gts in zip(pred_list, gt_list):
        tp, fp, fn = match_detections(preds, gts, iou_thresh)
        all_tp += tp; all_fp += fp; all_fn += fn
    p, r, _ = precision_recall_f1(all_tp, all_fp, all_fn)
    return p * r / (p + r + 1e-6) * 2


def annotation_time_reduction(manual_seconds_per_image: float,
                               auto_seconds_per_image: float,
                               correction_rate: float = 0.15) -> dict:
    corrected = auto_seconds_per_image + manual_seconds_per_image * correction_rate
    reduction = (manual_seconds_per_image - corrected) / manual_seconds_per_image * 100
    return {
        "manual_time_s":         manual_seconds_per_image,
        "auto_time_s":           auto_seconds_per_image,
        "hybrid_time_s":         corrected,
        "reduction_percent":     round(reduction, 1),
    }


def compute_pixel_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union        = np.logical_or(pred_mask, gt_mask).sum()
    return float(intersection / (union + 1e-6))
