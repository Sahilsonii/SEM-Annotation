import os
import json
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import build_model


class PatchDataset(Dataset):
    def __init__(self, image_paths, label_paths, patch_size=64):
        self.samples    = []
        self.patch_size = patch_size
        for img_path, lbl_path in zip(image_paths, label_paths):
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if gray is None or not os.path.exists(lbl_path):
                continue
            h, w = gray.shape
            with open(lbl_path) as f:
                lines = [l.strip() for l in f if l.strip()]
            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue
                xc = float(parts[1]) * w
                yc = float(parts[2]) * h
                bw = float(parts[3]) * w
                bh = float(parts[4]) * h
                x  = int(xc - bw / 2)
                y  = int(yc - bh / 2)
                roi = gray[max(0,y):y+int(bh), max(0,x):x+int(bw)]
                if roi.size == 0:
                    continue
                roi = cv2.resize(roi, (patch_size, patch_size))
                self.samples.append((roi, 1))
            for _ in range(max(1, len(lines))):
                rx = np.random.randint(0, max(1, w - patch_size))
                ry = np.random.randint(0, max(1, h - patch_size))
                neg = gray[ry:ry+patch_size, rx:rx+patch_size]
                if neg.shape == (patch_size, patch_size):
                    self.samples.append((neg, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        return tensor, torch.tensor(label, dtype=torch.long)


def train_classifier(model_name, image_paths, label_paths, output_dir,
                     epochs=10, lr=1e-3, batch_size=16):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PatchDataset(image_paths, label_paths)
    if len(dataset) == 0:
        return {"error": "No samples found"}

    split   = int(0.8 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [split, len(dataset)-split])
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model     = build_model(model_name, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_accs = [], []
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for imgs, labels in train_ld:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
        train_losses.append(running / max(1, len(train_ld)))

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_ld:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        val_accs.append(correct / max(1, total))
        print(f"  [{model_name}] Epoch {epoch}/{epochs}  loss={train_losses[-1]:.4f}  val_acc={val_accs[-1]:.3f}")

    elapsed = time.perf_counter() - t0
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, f"{model_name}_weights.pt"))
    return {
        "model": model_name,
        "epochs": epochs,
        "train_time_s": round(elapsed, 2),
        "final_val_acc": round(val_accs[-1], 4),
        "train_losses": train_losses,
        "val_accs": val_accs,
    }


def plot_benchmark_comparison(results_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    names    = [r["model"]          for r in results_list]
    accs     = [r["final_val_acc"]  for r in results_list]
    times    = [r["train_time_s"]   for r in results_list]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    palette   = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2"]

    axes[0].bar(names, accs, color=palette[:len(names)], edgecolor="black", linewidth=0.7)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Validation Accuracy")
    axes[0].set_title("Model Benchmark: Accuracy")
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    axes[1].bar(names, times, color=palette[:len(names)], edgecolor="black", linewidth=0.7)
    axes[1].set_ylabel("Training Time (s)")
    axes[1].set_title("Model Benchmark: Training Time")
    for i, v in enumerate(times):
        axes[1].text(i, v + 0.5, f"{v:.1f}s", ha="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "benchmark_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Benchmark plot saved: {path}")

    with open(os.path.join(output_dir, "benchmark_results.json"), "w") as f:
        out = [{k: v for k, v in r.items() if k not in ("train_losses", "val_accs")} for r in results_list]
        json.dump(out, f, indent=2)
