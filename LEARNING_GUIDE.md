# Comprehensive Learning Guide

This document consolidates all project documentation for learning purposes.



<!-- ============================== -->
# SOURCE: PROJECT_STRUCTURE.md
<!-- ============================== -->

# Project Structure

This document describes the organization of the SEM-Annotation project.

## Directory Layout

```
SEM-Annotation/
├── app.py                      # Main Streamlit application entry point
├── requirements.txt            # Python dependencies
├── LICENSE                     # Project license
├── README.md                   # Project documentation
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── model_handler.py        # Model training and inference handlers
│   ├── logger_setup.py         # Logging configuration
│   ├── handlers/               # Annotation and auto-labeling handlers
│   │   ├── __init__.py
│   │   ├── canvas_ui.py        # Canvas annotation interface
│   │   ├── detectron2_handler.py  # Detectron2 auto-annotation
│   │   ├── opencv_handler.py   # OpenCV-based auto-annotation
│   │   ├── sam_handler.py      # SAM (Segment Anything) handler
│   │   └── utils.py            # Utility functions for handlers
│   └── utils/                  # General utility modules
│
├── scripts/                    # Standalone scripts and tools
│   ├── split_dataset.py        # Dataset splitting and patching script
│   └── train_all_gpu.py        # Multi-GPU training script
│
├── data/                       # Data directory
│   ├── raw/                    # Raw, unprocessed data
│   ├── processed/              # Processed and cleaned data
│   └── labels/                 # Generated labels
│
├── balanced_dataset/           # Balanced dataset for training
│   ├── data.yaml              # YOLO dataset configuration
│   ├── images/                # Balanced image set
│   └── labels/                # Corresponding labels
│
├── labels/                     # Original labeled data by category
│   ├── 3D perovskite/
│   ├── 3D perovskite with PbI2 excess/
│   ├── 3D perovskite with pinholes/
│   ├── 3D-2D mixed perovskite/
│   └── 3D-2D mixed perovskite with pinholes/
│
├── yolo_dataset/              # YOLO format dataset
│
├── configs/                   # Configuration files
│
├── tests/                     # Unit and integration tests
│
├── docs/                      # Additional documentation
│
├── .streamlit/                # Streamlit configuration
├── .vscode/                   # VS Code settings
└── env/                       # Python virtual environment (not tracked)
```

## Module Organization

### src/
Contains the main source code for the application:
- **model_handler.py**: Handles model training, inference, and management
- **logger_setup.py**: Configures logging for the application
- **handlers/**: Specialized handlers for different annotation methods
  - Canvas UI for manual annotation
  - Auto-annotation using Detectron2, OpenCV, and SAM

### scripts/
Standalone scripts that can be run independently:
- **split_dataset.py**: Creates patches and prepares datasets for training
- **train_all_gpu.py**: Multi-GPU training orchestration

### Import Structure

After restructuring, imports should follow this pattern:

```python
# In app.py or other root-level files
from src.model_handler import ModelHandler
from src.logger_setup import setup_logger
from src.handlers import annotation_interface, auto_annotate_with_sam

# In scripts
# Scripts can import from src using:
import sys
sys.path.append('../')  # or os.path.abspath('..')
from src.model_handler import ModelHandler
```

## Changes Made

### Deleted Files
The following redundant files were removed after reorganization:
- Root-level Python files: `balance_dataset.py`, `logger_setup.py`, `model_handler.py`, `train_all_gpu.py`
- Duplicate handler files in `balanced_dataset/`: `canvas_ui.py`, `detectron2_handler.py`, `opencv_handler.py`, `sam_handler.py`, `utils.py`
- Empty template directory: `work/`

### Files Moved
- Handler modules → `src/handlers/`
- Core modules → `src/`
- Utility scripts → `scripts/`

### Updated Imports
- `app.py`: Updated to import from `src.model_handler`

## Benefits of This Structure

1. **Clear Separation**: Source code (`src/`) separated from scripts (`scripts/`)
2. **Modular**: Handlers are grouped together in `src/handlers/`
3. **Scalable**: Easy to add new modules and handlers
4. **Standard**: Follows Python project conventions
5. **Clean**: No duplicate files or redundant directories

## Next Steps

To add new functionality:
1. Add source code modules to `src/`
2. Add standalone scripts to `scripts/`
3. Add tests to `tests/`
4. Update `__init__.py` files to expose new functionality


<!-- ============================== -->
