"""
SAM (Segment Anything Model) handler for auto-annotation.
"""
import os
import logging
from PIL import Image

logger = logging.getLogger(__name__)


def auto_annotate_with_sam(image_list, labels_dir, conf_threshold=0.5):
    """
    Auto-annotate images using SAM (Segment Anything Model).
    
    Args:
        image_list: List of image paths
        labels_dir: Directory to save labels
        conf_threshold: Confidence threshold for detections
        
    Returns:
        Number of images annotated
    """
    try:
        # Try to import SAM
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError:
        raise ImportError(
            "SAM not installed. Please install with: pip install segment-anything"
        )
    
    os.makedirs(labels_dir, exist_ok=True)
    
    # Initialize SAM model
    # Note: This requires downloading SAM checkpoint
    logger.info("Loading SAM model...")
    sam_checkpoint = "sam_vit_h_4b8939.pth"  # Default SAM checkpoint
    model_type = "vit_h"
    
    if not os.path.exists(sam_checkpoint):
        logger.error(f"SAM checkpoint not found: {sam_checkpoint}")
        logger.info("Download from: https://github.com/facebookresearch/segment-anything")
        return 0
    
    try:
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        mask_generator = SamAutomaticMaskGenerator(sam)
    except Exception as e:
        logger.error(f"Failed to load SAM model: {e}")
        return 0
    
    count = 0
    for img_path in image_list:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(labels_dir, base_name + ".txt")
        
        if os.path.exists(txt_path):
            logger.info(f"Skipping {img_path} - already annotated")
            continue
        
        try:
            # Load image
            img = Image.open(img_path).convert("RGB")
            import numpy as np
            img_array = np.array(img)
            
            # Generate masks
            masks = mask_generator.generate(img_array)
            
            # Convert to YOLO format
            h, w = img_array.shape[:2]
            yolo_lines = []
            
            for mask in masks:
                if mask['predicted_iou'] < conf_threshold:
                    continue
                
                # Get bounding box from segmentation
                segmentation = mask['segmentation']
                y_indices, x_indices = np.where(segmentation)
                
                if len(x_indices) == 0 or len(y_indices) == 0:
                    continue
                
                x_min, x_max = x_indices.min(), x_indices.max()
                y_min, y_max = y_indices.min(), y_indices.max()
                
                # Convert to YOLO format
                x_center = ((x_min + x_max) / 2) / w
                y_center = ((y_min + y_max) / 2) / h
                width = (x_max - x_min) / w
                height = (y_max - y_min) / h
                
                # Class 1 (pinholes) as default
                yolo_lines.append(f"1 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # Save annotations
            with open(txt_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
            
            count += 1
            logger.info(f"Annotated {img_path}: {len(yolo_lines)} objects detected")
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
    
    return count
