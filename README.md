# SEM Annotation – GPU Lab Training

This repository has been cleaned and configured for headless batch training on a dedicated GPU machine. It contains a complete sequential training pipeline for 6 YOLO variants (`yolov8s, yolov8m, yolov8l, yolo26s, yolo26m, yolo26l`) on a balanced dataset.

## 🚀 Setup Instructions

1. **Clone the repository** to your GPU Lab machine.
2. **Create a new Virtual Environment** (Python 3.10+ recommended):
   ```bash
   python -m venv env
   
   # Windows
   .\env\Scripts\activate
   
   # Linux/Mac
   source env/bin/activate
   ```
3. **Install Dependencies**:
   The `requirements.txt` contains the exact library versions mapped to CUDA 11.8.
   ```bash
   pip install -r requirements.txt
   ```
   > **Note on PyTorch**: If your GPU lab requires a different CUDA version (e.g., CUDA 12.1), you may need to install PyTorch manually from the [official site](https://pytorch.org/get-started/locally/) before running the requirements file.

## 🧠 Running the Benchmark

The `train_all_gpu.py` script handles everything sequentially.

```bash
python train_all_gpu.py
```

**What it does:**
- Verifies the `balanced_dataset` is configured correctly.
- Downloads any missing pre-trained weights (`yolov8l.pt`, `yolo26l.pt`, etc.) automatically.
- Trains each of the 6 models for 200 epochs.
- Saves the best weights (`best.pt`) inside `runs/detect/<model_name>_<timestamp>/weights/`.
- Automatically evaluates the validation set and dumps detailed metrics (mAP50, mAP50-95, precision, recall, F1, training time) into JSON files inside `runs/metrics/`.

## 📊 Visualizing Results

Once training finishes, you can use the Streamlit app's built-in **Model Comparison** dashboard to view and chart the results side-by-side.

```bash
streamlit run app.py
```
- Expand the sidebar and select **Model Comparison**.
- The dashboard will automatically read the JSON files in `runs/metrics/`, highlight the best model overall, and allow you to load it for the auto-annotation feature.

---

### Clean Repository Info
- The `env/` folder currently here is specific to the development machine and should not be tracked by git.
- The `runs/` folder is intentionally omitted from tracking. New runs and metrics will populate automatically.
