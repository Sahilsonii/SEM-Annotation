"""
OpenCV-based auto-annotation handler for defect detection.
"""
import cv2
import numpy as np
import os
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def auto_annotate_with_opencv(
    image_list,
    labels_dir,
    class_id=1,
    method="threshold",
    threshold1=80,
    threshold2=255,
    brightness=0,
    contrast=1.0,
    min_area=50,
    max_area=10000,
    use_clahe=False,
    clahe_clip=2.0,
    clahe_grid=8,
    overwrite=False
):
    """
    Auto-annotate images using OpenCV methods.
    
    Args:
        image_list: List of image paths
        labels_dir: Directory to save labels
        class_id: Class ID for detected objects
        method: Detection method (threshold, tophat_bright, etc.)
        threshold1: Primary threshold value
        threshold2: Secondary threshold value
        brightness: Brightness adjustment (-100 to 100)
        contrast: Contrast adjustment (0.5 to 3.0)
        min_area: Minimum contour area
        max_area: Maximum contour area
        use_clahe: Enable CLAHE preprocessing
        clahe_clip: CLAHE clip limit
        clahe_grid: CLAHE grid size
        overwrite: Overwrite existing labels
        
    Returns:
        Number of images annotated
    """
    os.makedirs(labels_dir, exist_ok=True)
    count = 0
    
    for img_path in image_list:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(labels_dir, base_name + ".txt")
        
        if os.path.exists(txt_path) and not overwrite:
            logger.info(f"Skipping {img_path} - already annotated")
            continue
        
        try:
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                logger.error(f"Failed to load {img_path}")
                continue
            
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply CLAHE if enabled
            if use_clahe:
                clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
                gray = clahe.apply(gray)
            
            # Adjust brightness and contrast
            if brightness != 0 or contrast != 1.0:
                gray = cv2.convertScaleAbs(gray, alpha=contrast, beta=brightness)
            
            # Apply detection method
            if method == "threshold":
                _, binary = cv2.threshold(gray, threshold1, 255, cv2.THRESH_BINARY_INV)
            elif method == "threshold_bright":
                _, binary = cv2.threshold(gray, threshold1, 255, cv2.THRESH_BINARY)
            elif method == "tophat_bright":
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
                _, binary = cv2.threshold(tophat, threshold1, 255, cv2.THRESH_BINARY)
            elif method == "color_mask":
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                lower = np.array([0, 0, 0])
                upper = np.array([180, 255, threshold1])
                binary = cv2.inRange(hsv, lower, upper)
            elif method == "color_mask_bright":
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                lower = np.array([0, 0, threshold1])
                upper = np.array([180, 255, 255])
                binary = cv2.inRange(hsv, lower, upper)
            elif method == "adaptive":
                binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY_INV, 11, 2)
            elif method == "canny":
                binary = cv2.Canny(gray, threshold1, threshold2)
            elif method == "watershed":
                _, binary = cv2.threshold(gray, threshold1, 255, cv2.THRESH_BINARY_INV)
                kernel = np.ones((3, 3), np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
            else:
                _, binary = cv2.threshold(gray, threshold1, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter and convert to YOLO format
            yolo_lines = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area <= area <= max_area:
                    x, y, cw, ch = cv2.boundingRect(contour)
                    x_center = (x + cw / 2) / w
                    y_center = (y + ch / 2) / h
                    w_norm = cw / w
                    h_norm = ch / h
                    yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
            
            # Save annotations
            with open(txt_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
            
            count += 1
            logger.info(f"Annotated {img_path}: {len(yolo_lines)} objects detected")
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
    
    return count
