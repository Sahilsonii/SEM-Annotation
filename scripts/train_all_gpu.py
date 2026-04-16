"""
train_all_gpu.py

Script for headless sequential training of all 6 required YOLO models
on a dedicated GPU machine, plus Faster R-CNN (torchvision).

Run this script: python train_all_gpu.py
"""
import os
import sys
import json
import time
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

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

# Faster R-CNN config
FRCNN_EPOCHS = 50
FRCNN_BATCH = 4
FRCNN_LR = 0.005
FRCNN_MOMENTUM = 0.9
FRCNN_WEIGHT_DECAY = 0.0005
FRCNN_LR_STEP_SIZE = 15
FRCNN_LR_GAMMA = 0.1
FRCNN_NUM_CLASSES = 6  # 5 classes + 1 background
FRCNN_IMGSZ = 800
FRCNN_VAL_SPLIT = 0.2
# ─────────────────────────────────────────────────────────────────────────────

HERE         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
BALANCED_DIR = os.path.join(PROJECT_ROOT, "balanced_dataset")
SPLIT_DIR    = os.path.join(PROJECT_ROOT, "balanced_dataset_split")
YAML_PATH    = os.path.join(BALANCED_DIR, "data.yaml")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, HERE)


# ── YOLO TRAINING (UNCHANGED LOGIC) ──────────────────────────────────────────

def train_yolo_models():
    if not os.path.exists(YAML_PATH):
        logger.error(f"data.yaml not found at {YAML_PATH}")
        logger.error("Ensure the balanced dataset is present before training.")
        return []

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

    try:
        from model_handler import ModelHandler
    except ImportError:
        try:
            from src.model_handler import ModelHandler
        except ImportError:
            logger.error("Could not import ModelHandler.")
            return []

    print(f"\n{'='*60}")
    print(f"  GPU Lab Batch Training: {len(MODELS_TO_TRAIN)} YOLO models × {EPOCHS} epochs")
    print(f"{'='*60}\n")

    results_summary = []

    for i, m in enumerate(MODELS_TO_TRAIN, 1):
        label   = m["label"]
        weights = m["weights"]
        wpath   = os.path.join(HERE, weights)

        print(f"\n[{i}/{len(MODELS_TO_TRAIN)}] ▶ Training {label}  (pre-trained weights: {weights})")
        print("-" * 60)

        try:
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

    return results_summary


# ── FASTER R-CNN TRAINING ─────────────────────────────────────────────────────

def train_faster_rcnn():
    import torch
    import torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torch.utils.data import DataLoader, random_split

    from src.datasets.frcnn_dataset import SEMFasterRCNNDataset, collate_fn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Faster R-CNN training on: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")

    exp_dir = os.path.join(PROJECT_ROOT, "experiments", "faster_rcnn")
    weights_dir = os.path.join(exp_dir, "weights")
    logs_dir = os.path.join(exp_dir, "logs")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    data_root = SPLIT_DIR if os.path.isdir(SPLIT_DIR) else BALANCED_DIR
    img_dir = os.path.join(data_root, "images")
    lbl_dir = os.path.join(data_root, "labels")

    logger.info(f"Dataset root: {data_root}")

    full_dataset = SEMFasterRCNNDataset(
        img_dir=img_dir,
        lbl_dir=lbl_dir,
        num_classes=FRCNN_NUM_CLASSES,
    )

    if len(full_dataset) == 0:
        logger.error("No images found for Faster R-CNN training.")
        return "❌ No data", "—"

    logger.info(f"Total samples: {len(full_dataset)}")

    val_size = max(1, int(len(full_dataset) * FRCNN_VAL_SPLIT))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=FRCNN_BATCH, shuffle=True,
        num_workers=0, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=FRCNN_BATCH, shuffle=False,
        num_workers=0, collate_fn=collate_fn, pin_memory=True,
    )

    logger.info(f"Train: {train_size} | Val: {val_size}")

    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, FRCNN_NUM_CLASSES)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=FRCNN_LR, momentum=FRCNN_MOMENTUM,
        weight_decay=FRCNN_WEIGHT_DECAY,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=FRCNN_LR_STEP_SIZE, gamma=FRCNN_LR_GAMMA,
    )

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_epoch = 0

    print(f"\n{'='*60}")
    print(f"  Faster R-CNN Training: {FRCNN_EPOCHS} epochs")
    print(f"  Batch: {FRCNN_BATCH} | LR: {FRCNN_LR} | ImgSz: {FRCNN_IMGSZ}")
    print(f"{'='*60}\n")

    t_start = time.time()

    for epoch in range(1, FRCNN_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            valid_batch = []
            valid_targets = []
            for img, tgt in zip(images, targets):
                if tgt["boxes"].numel() > 0:
                    valid_batch.append(img)
                    valid_targets.append(tgt)

            if len(valid_batch) == 0:
                continue

            loss_dict = model(valid_batch, valid_targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
            optimizer.step()

            epoch_loss += losses.item()
            num_batches += 1

        lr_scheduler.step()

        avg_train_loss = epoch_loss / max(num_batches, 1)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            model.train()
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                valid_batch = []
                valid_targets = []
                for img, tgt in zip(images, targets):
                    if tgt["boxes"].numel() > 0:
                        valid_batch.append(img)
                        valid_targets.append(tgt)

                if len(valid_batch) == 0:
                    continue

                loss_dict = model(valid_batch, valid_targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
                val_batches += 1
            model.eval()

        avg_val_loss = val_loss / max(val_batches, 1)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_path = os.path.join(weights_dir, "best.pt")
            torch.save(model.state_dict(), best_path)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{FRCNN_EPOCHS} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    total_time = time.time() - t_start

    last_path = os.path.join(weights_dir, "last.pt")
    torch.save(model.state_dict(), last_path)

    mAP = _compute_simple_map(model, val_loader, device)

    metrics = {
        "model": "faster_rcnn_resnet50_fpn",
        "timestamp": datetime.now().isoformat(),
        "epochs": FRCNN_EPOCHS,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "final_train_loss": round(train_losses[-1], 6) if train_losses else 0,
        "final_val_loss": round(val_losses[-1], 6) if val_losses else 0,
        "mAP50": round(mAP, 4),
        "train_time_s": round(total_time, 1),
        "num_classes": FRCNN_NUM_CLASSES,
        "train_samples": train_size,
        "val_samples": val_size,
        "batch_size": FRCNN_BATCH,
        "learning_rate": FRCNN_LR,
        "train_losses": [round(l, 6) for l in train_losses],
        "val_losses": [round(l, 6) for l in val_losses],
        "weights_best": best_path,
        "weights_last": last_path,
    }

    metrics_path = os.path.join(exp_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    log_path = os.path.join(logs_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(log_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  ✅ Faster R-CNN training complete in {total_time:.1f}s")
    print(f"  📊 Best val loss: {best_val_loss:.4f} (epoch {best_epoch})")
    print(f"  📊 mAP@50: {mAP:.4f}")
    print(f"  💾 Weights: {weights_dir}")
    print(f"  📄 Metrics: {metrics_path}")

    return "✅ Success", best_path


def _compute_simple_map(model, val_loader, device, iou_threshold=0.5):
    import torch
    model.eval()
    all_tp = 0
    all_fp = 0
    all_fn = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            predictions = model(images)

            for pred, tgt in zip(predictions, targets):
                pred_boxes = pred["boxes"].cpu()
                pred_scores = pred["scores"].cpu()
                pred_labels = pred["labels"].cpu()
                gt_boxes = tgt["boxes"]
                gt_labels = tgt["labels"]

                score_mask = pred_scores > 0.5
                pred_boxes = pred_boxes[score_mask]
                pred_labels = pred_labels[score_mask]

                if len(gt_boxes) == 0:
                    all_fp += len(pred_boxes)
                    continue
                if len(pred_boxes) == 0:
                    all_fn += len(gt_boxes)
                    continue

                gt_matched = set()
                for pb, pl in zip(pred_boxes, pred_labels):
                    best_iou = 0
                    best_idx = -1
                    for gi, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                        if gi in gt_matched:
                            continue
                        if int(pl) != int(gl):
                            continue
                        iou = _box_iou_single(pb, gb)
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = gi
                    if best_iou >= iou_threshold and best_idx >= 0:
                        all_tp += 1
                        gt_matched.add(best_idx)
                    else:
                        all_fp += 1
                all_fn += len(gt_boxes) - len(gt_matched)

    precision = all_tp / max(all_tp + all_fp, 1)
    recall = all_tp / max(all_tp + all_fn, 1)
    mAP = 2 * precision * recall / max(precision + recall, 1e-6)
    return mAP


def _box_iou_single(box1, box2):
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (float(box1[2]) - float(box1[0])) * (float(box1[3]) - float(box1[1]))
    area2 = (float(box2[2]) - float(box2[0])) * (float(box2[3]) - float(box2[1]))
    union = area1 + area2 - inter
    return inter / max(union, 1e-6)


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train YOLO + Faster R-CNN models")
    parser.add_argument("--yolo-only", action="store_true", help="Train only YOLO models")
    parser.add_argument("--frcnn-only", action="store_true", help="Train only Faster R-CNN")
    parser.add_argument("--frcnn-epochs", type=int, default=None)
    parser.add_argument("--frcnn-batch", type=int, default=None)
    parser.add_argument("--frcnn-lr", type=float, default=None)
    args = parser.parse_args()

    if args.frcnn_epochs:
        FRCNN_EPOCHS = args.frcnn_epochs
    if args.frcnn_batch:
        FRCNN_BATCH = args.frcnn_batch
    if args.frcnn_lr:
        FRCNN_LR = args.frcnn_lr

    results_summary = []

    # ── YOLO ──
    if not args.frcnn_only:
        yolo_results = train_yolo_models()
        results_summary.extend(yolo_results)

    # ── Faster R-CNN ──
    if not args.yolo_only:
        print(f"\n{'='*60}")
        print(f"  STARTING FASTER R-CNN TRAINING")
        print(f"{'='*60}")
        try:
            frcnn_status, frcnn_path = train_faster_rcnn()
            results_summary.append(("faster_rcnn", frcnn_status, frcnn_path))
        except Exception as e:
            print(f"  ❌ Faster R-CNN failed: {e}")
            import traceback
            traceback.print_exc()
            results_summary.append(("faster_rcnn", f"❌ Failed: {e}", "—"))

    # ── Summary ──
    print(f"\n{'='*60}")
    print("  ALL TRAINING RUNS COMPLETE")
    print(f"{'='*60}")
    for label, status, pt in results_summary:
        print(f"  {label:16s} | {status}")
    print()
