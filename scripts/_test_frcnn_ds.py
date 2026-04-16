import sys
sys.path.insert(0, ".")
from src.datasets.frcnn_dataset import SEMFasterRCNNDataset, collate_fn

ds = SEMFasterRCNNDataset("balanced_dataset/images", "balanced_dataset/labels")
print(f"Dataset loaded: {len(ds)} samples")

img, tgt = ds[0]
print(f"Image shape: {img.shape}")
boxes = tgt["boxes"]
labels = tgt["labels"]
print(f"Boxes shape: {boxes.shape}")
print(f"Labels: {labels}")
print(f"Area: {tgt['area']}")
print("FRCNN Dataset OK")
