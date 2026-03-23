# SEM Annotation – Perovskite Image Annotation & Training

A comprehensive tool for annotating SEM (Scanning Electron Microscopy) images of perovskite materials and training YOLO models for automated detection. This repository includes a Streamlit-based annotation interface, multiple auto-annotation methods (SAM, Detectron2, OpenCV), and a complete training pipeline for GPU-accelerated model training.

## ✨ Features

- **Interactive Annotation Interface**: Streamlit-based canvas UI for manual annotation
- **Auto-Annotation Tools**:
  - SAM (Segment Anything Model)
  - Detectron2
  - OpenCV-based segmentation
- **Dataset Management**: Balanced dataset creation and YOLO format conversion
- **Multi-Model Training**: Support for 6 YOLO variants (`yolov8s, yolov8m, yolov8l, yolo26s, yolo26m, yolo26l`)
- **Model Comparison Dashboard**: Visual comparison of trained models with metrics
- **GPU-Accelerated**: Optimized for CUDA 11.8+ GPU training

---

## 📁 Project Structure

```
SEM-Annotation/
├── app.py                      # Main Streamlit application
├── src/                        # Source code modules
│   ├── handlers/               # Annotation handlers
│   │   ├── canvas_ui.py       # Manual annotation interface
│   │   ├── sam_handler.py     # SAM auto-annotation
│   │   ├── detectron2_handler.py  # Detectron2 integration
│   │   └── opencv_handler.py  # OpenCV segmentation
│   ├── model_handler.py        # Model training & inference
│   └── logger_setup.py         # Logging configuration
├── scripts/                    # Utility scripts
│   ├── balance_dataset.py     # Dataset balancing tool
│   └── train_all_gpu.py       # Multi-GPU training pipeline
├── balanced_dataset/           # Training-ready dataset
│   ├── images/                # Balanced image set
│   └── labels/                # YOLO format labels
├── data/                       # Data directory
│   ├── raw/                   # Original data
│   ├── processed/             # Processed data
│   └── labels/                # Generated labels
└── configs/                    # Configuration files
```

> 📖 For detailed structure documentation, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

---

## 🚀 Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd SEM-Annotation
```

### 2. Create Virtual Environment
Python 3.10+ recommended:
```bash
python -m venv env

# Windows
.\env\Scripts\activate

# Linux/Mac
source env/bin/activate
```

### 3. Install Dependencies
The `requirements.txt` contains library versions mapped to CUDA 11.8:
```bash
pip install -r requirements.txt
```

> **Note on PyTorch**: If your GPU requires a different CUDA version (e.g., CUDA 12.1), install PyTorch manually from the [official site](https://pytorch.org/get-started/locally/) before installing other requirements.

---

## 🎯 Usage

### Interactive Annotation Interface
Launch the Streamlit app for manual annotation:
```bash
streamlit run app.py
```

Features available in the interface:
- **Manual Annotation**: Draw bounding boxes on SEM images
- **Auto-Annotation**: Use SAM, Detectron2, or OpenCV for automated labeling
- **Dataset Balancing**: Create balanced training datasets
- **Model Training**: Train YOLO models directly from the UI
- **Model Comparison**: Compare performance metrics across models

### Batch Training (GPU Lab)
For headless batch training on dedicated GPU machines:

```bash
python scripts/train_all_gpu.py
```

**What it does:**
- Verifies `balanced_dataset/` configuration
- Downloads missing pre-trained YOLO weights automatically
- Trains 6 YOLO models sequentially (200 epochs each)
- Saves best weights to `runs/detect/<model>_<timestamp>/weights/best.pt`
- Generates detailed metrics (mAP50, mAP50-95, precision, recall, F1) in `runs/metrics/`

### Dataset Balancing
Create a balanced dataset from labeled images:
```bash
python scripts/balance_dataset.py
```

---

## 📊 Model Comparison & Visualization

After training, view results in the Streamlit dashboard:

1. Launch the app: `streamlit run app.py`
2. Navigate to **Model Comparison** in the sidebar
3. View metrics, charts, and load the best model

The dashboard automatically reads JSON metrics from `runs/metrics/` and highlights the best-performing model.

---

## 🔧 Development

### Project Organization
- **Source Code**: All modules in `src/` directory
- **Scripts**: Standalone tools in `scripts/` directory
- **Tests**: Unit tests in `tests/` directory
- **Documentation**: Additional docs in `docs/` directory

### Import Structure
```python
# In app.py or root-level scripts
from src.model_handler import ModelHandler
from src.handlers import annotation_interface, auto_annotate_with_sam
```

### Adding New Features
1. Add modules to `src/` for core functionality
2. Add scripts to `scripts/` for standalone tools
3. Update `__init__.py` files to expose new functionality
4. Add tests to `tests/` directory

---

## 📝 Supported Perovskite Categories

- 3D perovskite
- 3D perovskite with PbI2 excess
- 3D perovskite with pinholes
- 3D-2D mixed perovskite
- 3D-2D mixed perovskite with pinholes

---

## 🛠️ Technical Details

- **Python**: 3.10+
- **Deep Learning**: PyTorch, Ultralytics YOLO
- **Computer Vision**: OpenCV, Detectron2, SAM
- **UI Framework**: Streamlit
- **GPU**: CUDA 11.8+ compatible
- **Data Format**: YOLO format (`.txt` annotations)

---

## 📦 Repository Notes

- `env/` folder is environment-specific and not tracked by git
- `runs/` folder is generated during training and not tracked
- Training metrics are saved to `runs/metrics/` as JSON files
- Model weights are saved to `runs/detect/<model>/weights/`

---

## 📄 License

See [LICENSE](LICENSE) file for details.
