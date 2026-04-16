import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class SEMFasterRCNNDataset(Dataset):

    def __init__(self, img_dir, lbl_dir, num_classes=6, transforms=None):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.num_classes = num_classes
        self.transforms = transforms
        self.extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        self.samples = []
        self._scan_directories()

    def _scan_directories(self):
        if not os.path.isdir(self.img_dir):
            return
        for root, _, files in os.walk(self.img_dir):
            for fname in sorted(files):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in self.extensions:
                    continue
                img_path = os.path.join(root, fname)
                rel = os.path.relpath(root, self.img_dir)
                lbl_path = os.path.join(self.lbl_dir, rel, os.path.splitext(fname)[0] + ".txt")
                if not os.path.exists(lbl_path):
                    lbl_path_flat = os.path.join(self.lbl_dir, os.path.splitext(fname)[0] + ".txt")
                    if os.path.exists(lbl_path_flat):
                        lbl_path = lbl_path_flat
                self.samples.append((img_path, lbl_path))

    def __len__(self):
        return len(self.samples)

    def _parse_yolo_labels(self, lbl_path, img_w, img_h):
        boxes = []
        labels = []
        if not os.path.exists(lbl_path):
            return boxes, labels
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                xc = float(parts[1]) * img_w
                yc = float(parts[2]) * img_h
                bw = float(parts[3]) * img_w
                bh = float(parts[4]) * img_h
                x1 = xc - bw / 2.0
                y1 = yc - bh / 2.0
                x2 = xc + bw / 2.0
                y2 = yc + bh / 2.0
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_w, x2)
                y2 = min(img_h, y2)
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls_id + 1)
        return boxes, labels

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        boxes, labels = self._parse_yolo_labels(lbl_path, img_w, img_h)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }

        img_tensor = torch.from_numpy(
            np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        )

        if self.transforms is not None:
            img_tensor = self.transforms(img_tensor)

        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))
