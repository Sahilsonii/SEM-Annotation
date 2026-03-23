import os
import sys
import numpy as np
from PIL import Image
import cv2

# Add detectron2 path
detectron_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "detectron2_pipeline")
if detectron_path not in sys.path:
    sys.path.append(detectron_path)

try:
    import torch
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    DETECTRON2_AVAILABLE = True
except:
    DETECTRON2_AVAILABLE = False

def auto_annotate_with_detectron2(images, output_dir, model_path, conf_threshold=0.25):
    """Auto-annotate images using Detectron2"""
    if not DETECTRON2_AVAILABLE:
        raise ImportError("Detectron2 not available.")
    
    # Setup predictor
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_threshold
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 1024
    cfg.MODEL.DEVICE = "cpu"
    
    predictor = DefaultPredictor(cfg)
    labeled_count = 0
    
    for img_path in images:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(output_dir, base_name + ".txt")
        
        # Skip if already labeled
        if os.path.exists(label_path):
            continue
        
        # Load image
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        # Run inference
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")
        
        # Convert to YOLO format
        yolo_lines = []
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            
            # Convert to YOLO format
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h
            
            yolo_lines.append(f"1 {x_center} {y_center} {width} {height}")
        
        # Save if any detections
        if yolo_lines:
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
            labeled_count += 1
    
    return labeled_count
