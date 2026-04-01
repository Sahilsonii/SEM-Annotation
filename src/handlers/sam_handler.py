import os
import sys
import numpy as np
import cv2
import urllib.request

# Add SAM path
sam_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sam_pinhole_annotation")
if sam_path not in sys.path:
    sys.path.append(sam_path)

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except:
    SAM_AVAILABLE = False


def _download_sam_checkpoint(model_dir):
    """Download SAM checkpoint if not present"""
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, "sam_vit_b_01ec64.pth")
    
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    
    checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    
    print(f"Downloading SAM checkpoint from {checkpoint_url}")
    print("This is a large file (~375 MB) and may take several minutes...")
    
    try:
        urllib.request.urlretrieve(checkpoint_url, checkpoint_path)
        print(f"✓ Checkpoint downloaded to {checkpoint_path}")
        return checkpoint_path
    except Exception as e:
        raise RuntimeError(f"Failed to download SAM checkpoint: {e}")

def auto_annotate_with_sam(images, output_dir, conf_threshold=0.5):
    """Auto-annotate images using SAM with automatic mask generation"""
    if not SAM_AVAILABLE:
        raise ImportError("SAM not available. Install: pip install segment-anything")
    
    # Load SAM model with automatic download
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                              "sam_pinhole_annotation", "sam")
    
    # Download checkpoint if needed
    sam_checkpoint = _download_sam_checkpoint(model_dir)
    
    # Load model
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    sam.to(device="cpu")
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        pred_iou_thresh=conf_threshold,
        stability_score_thresh=conf_threshold
    )
    
    labeled_count = 0
    
    for img_path in images:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(output_dir, base_name + ".txt")
        
        # Skip if already labeled
        if os.path.exists(label_path):
            continue
        
        # Load image
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Generate masks
        masks = mask_generator.generate(img_rgb)
        
        # Convert to YOLO format
        yolo_lines = []
        for mask_data in masks:
            segmentation = mask_data['segmentation'].astype(np.uint8)
            
            # Get contour
            contours, _ = cv2.findContours(segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
                
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) < 10:
                continue
                
            points = largest_contour.squeeze(axis=1)
            if points.ndim != 2 or points.shape[0] < 3:
                continue
                
            # Convert to normalized YOLO format (class x1 y1 x2 y2 ...)
            norm_points = []
            for pt in points:
                nx = np.clip(float(pt[0]) / w, 0.0, 1.0)
                ny = np.clip(float(pt[1]) / h, 0.0, 1.0)
                norm_points.extend([f"{nx:.6f}", f"{ny:.6f}"])
            
            yolo_lines.append(f"1 " + " ".join(norm_points))
        
        # Save if any detections
        if yolo_lines:
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
            labeled_count += 1
    
    return labeled_count
