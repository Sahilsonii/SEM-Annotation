import os
import json
import shutil
import random
import logging
from datetime import datetime
from PIL import Image
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Absolute directory where this file lives (i.e. sem_app/).
# Used to anchor YOLO's project path so runs always land in sem_app/runs/detect/
# regardless of what the working directory is when the app is started.
_HERE = os.path.dirname(os.path.abspath(__file__))


class ModelHandler:
    def __init__(self, model_path: str = None):
        """
        Load a YOLO model.
        Args:
            model_path: Path to a trained .pt file or model name (e.g., 'yolo11s.pt').
        """
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            logger.info(f"Loaded model from: {model_path}")
        elif model_path:
            # Model name provided but file doesn't exist - download from Ultralytics
            self.model = YOLO(model_path)
            logger.info(f"Loaded pretrained {model_path}")
        else:
            self.model = YOLO("yolov8s.pt")
            logger.info("Loaded pretrained yolov8s.")

    # ------------------------------------------------------------------
    # Metrics saving
    # ------------------------------------------------------------------

    def save_metrics(
        self,
        model_name: str,
        val_results,
        weights_path: str,
        epochs: int,
        imgsz: int,
        train_time_s: float,
    ) -> str:
        """
        Run validation and save evaluation metrics to JSON.

        Args:
            model_name   : Human-readable label e.g. "yolov8s", "yolo26m".
            val_results  : Ultralytics Results object from model.val().
            weights_path : Absolute path to best.pt.
            epochs       : Number of training epochs completed.
            imgsz        : Training image size.
            train_time_s : Wall-clock training time in seconds.

        Returns:
            Absolute path to the written JSON file.
        """
        metrics_dir = os.path.join(_HERE, "runs", "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        # Ultralytics stores metrics under results.box for detection.
        box = val_results.box
        map50    = float(box.map50)    if hasattr(box, "map50")    else 0.0
        map5095  = float(box.map)      if hasattr(box, "map")      else 0.0
        precision= float(box.mp)       if hasattr(box, "mp")       else 0.0
        recall   = float(box.mr)       if hasattr(box, "mr")       else 0.0
        f1       = (2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0 else 0.0)

        record = {
            "model":          model_name,
            "timestamp":      datetime.now().isoformat(),
            "mAP50":          round(map50,    4),
            "mAP50_95":       round(map5095,  4),
            "precision":      round(precision, 4),
            "recall":         round(recall,   4),
            "f1":             round(f1,       4),
            "epochs":         epochs,
            "imgsz":          imgsz,
            "train_time_s":   round(train_time_s, 1),
            "weights_path":   weights_path,
        }

        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        out  = os.path.join(metrics_dir, f"{model_name}_{ts}.json")
        with open(out, "w") as f:
            json.dump(record, f, indent=2)

        logger.info(f"Metrics saved to: {out}")
        logger.info(
            f"  mAP50={map50:.4f}  mAP50-95={map5095:.4f}  "
            f"P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}"
        )
        return out

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_model(
        self,
        data_yaml_path: str,
        epochs: int = 10,
        imgsz: int = 640,
        model_name: str = None,
        **kwargs,
    ):
        """
        Train the YOLO model and save evaluation metrics.

        Args:
            data_yaml_path : Path to the dataset YAML file.
            epochs         : Number of training epochs.
            imgsz          : Input image size.
            model_name     : Label for this run (e.g. "yolov8s"). Used in the
                             metrics filename. Defaults to the weights stem.
            **kwargs       : Extra args forwarded to model.train()
                             (e.g. batch=16, patience=5, cos_lr=True).
        Returns:
            Tuple[results, metrics_path] — ultralytics results + path to JSON.
        """
        if not os.path.exists(data_yaml_path):
            raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")

        # Infer model_name from the loaded weights if not provided.
        if model_name is None:
            try:
                model_name = os.path.splitext(
                    os.path.basename(self.model.ckpt_path)
                )[0]
            except Exception:
                model_name = "yolo_model"

        # Timestamped run name — previous runs are never overwritten.
        run_name = kwargs.pop("name", f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        # FIX: absolute project path so runs never double-nest.
        project_path = os.path.join(_HERE, "runs", "detect")

        # FIX: explicit GPU if available.
        import torch
        import time
        device = 0 if torch.cuda.is_available() else "cpu"
        logger.info(
            f"Training on: "
            f"{'GPU — ' + torch.cuda.get_device_name(0) if device == 0 else 'CPU'}"
        )

        t0 = time.time()
        results = self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            project=project_path,
            name=run_name,
            device=device,
            **kwargs,
        )
        train_time_s = time.time() - t0

        # ---- Save metrics (run validation on best weights) ---------------
        best_pt = os.path.join(str(results.save_dir), "weights", "best.pt")
        metrics_path = None
        try:
            best_model  = YOLO(best_pt)
            val_results = best_model.val(data=data_yaml_path, imgsz=imgsz, device=device)
            metrics_path = self.save_metrics(
                model_name=model_name,
                val_results=val_results,
                weights_path=best_pt,
                epochs=epochs,
                imgsz=imgsz,
                train_time_s=train_time_s,
            )
        except Exception as e:
            logger.warning(f"Could not save metrics: {e}")

        return results, metrics_path

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, source, conf: float = 0.25, imgsz: int = 640):
        """
        Run inference on a source (file path, PIL Image, numpy array, etc.).
        """
        return self.model.predict(source, conf=conf, imgsz=imgsz, verbose=False)

    # ------------------------------------------------------------------
    # Dataset preparation
    # ------------------------------------------------------------------

    def prepare_data_and_yaml(
        self,
        image_label_pairs: list,
        project_root: str,
        classes_dict: dict,
        train_ratio: float = 0.8,
    ) -> str:
        """
        Shuffle, split, copy images+labels into a YOLO dataset tree, and write data.yaml.
        """
        if not image_label_pairs:
            raise ValueError("image_label_pairs is empty — nothing to prepare.")

        valid_pairs = []
        for img_p, lbl_p in image_label_pairs:
            if not os.path.exists(img_p):
                logger.warning(f"Image not found, skipping: {img_p}")
                continue
            if not os.path.exists(lbl_p):
                logger.warning(f"Label not found, skipping: {lbl_p}")
                continue
            valid_pairs.append((img_p, lbl_p))

        if not valid_pairs:
            raise ValueError("No valid (image, label) pairs found after validation.")

        random.shuffle(valid_pairs)
        split_idx   = max(1, min(int(len(valid_pairs) * train_ratio), len(valid_pairs) - 1))
        train_pairs = valid_pairs[:split_idx]
        val_pairs   = valid_pairs[split_idx:]
        logger.info(f"Dataset split — train: {len(train_pairs)}, val: {len(val_pairs)}")

        dataset_dir = os.path.join(project_root, "yolo_dataset")
        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)

        for split in ("train", "val"):
            os.makedirs(os.path.join(dataset_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(dataset_dir, "labels", split), exist_ok=True)

        for split, pairs in [("train", train_pairs), ("val", val_pairs)]:
            img_dir = os.path.join(dataset_dir, "images", split)
            lbl_dir = os.path.join(dataset_dir, "labels", split)
            for img_p, lbl_p in pairs:
                stem    = os.path.splitext(os.path.basename(img_p))[0]
                dst_img = os.path.join(img_dir, stem + ".jpg")
                dst_lbl = os.path.join(lbl_dir, stem + ".txt")
                try:
                    with Image.open(img_p) as img:
                        img.convert("RGB").save(dst_img, "JPEG")
                except Exception as e:
                    logger.error(f"Image conversion failed for {img_p}: {e} — skipping.")
                    continue
                shutil.copy(lbl_p, dst_lbl)

        names_list = [classes_dict[k] for k in sorted(classes_dict.keys())]
        names_yaml = "\n".join(f"  - {n}" for n in names_list)
        yaml_content = (
            f"path: {dataset_dir}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"nc: {len(classes_dict)}\n"
            f"names:\n{names_yaml}\n"
        )
        yaml_path = os.path.join(dataset_dir, "data.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        logger.info(f"data.yaml written to: {yaml_path}")
        return yaml_path

    # ------------------------------------------------------------------
    # Auto-annotation
    # ------------------------------------------------------------------

    def auto_annotate_folder(
        self,
        image_list: list,
        labels_dir: str,
        classes_map: dict = None,
        imgsz: int = 640,
        conf: float = 0.05,
    ) -> int:
        """
        Run inference on a list of images and save YOLO-format .txt annotations.
        """
        os.makedirs(labels_dir, exist_ok=True)

        written = 0
        for img_path in image_list:
            stem     = os.path.splitext(os.path.basename(img_path))[0]
            txt_path = os.path.join(labels_dir, stem + ".txt")

            if os.path.exists(txt_path):
                logger.debug(f"Already annotated, skipping: {img_path}")
                continue

            try:
                with Image.open(img_path) as img:
                    rgb_img = img.convert("RGB")
                results = self.predict(rgb_img, imgsz=imgsz, conf=conf)
            except Exception as e:
                logger.error(f"Inference failed for {img_path}: {e}")
                continue

            lines = []
            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                for box in r.boxes:
                    cls         = int(box.cls[0])
                    x, y, w, h  = box.xywhn[0].tolist()
                    lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

            with open(txt_path, "w") as f:
                f.write("\n".join(lines))

            written += 1
            status = f"{len(lines)} box(es)" if lines else "no detections"
            logger.info(f"Annotated {os.path.basename(img_path)}: {status}")

        logger.info(f"Auto-annotation complete. {written}/{len(image_list)} images written.")
        return written
