import os
import cv2
import numpy as np
from PIL import Image


def auto_annotate_with_opencv(
    images,
    output_dir,
    class_id=1,
    method="canny",
    threshold1=50,
    threshold2=150,
    brightness=0,
    contrast=1.0,
    min_area=100,
    max_area=10000,
    use_clahe=False,
    clahe_clip=2.0,
    clahe_grid=8,
    overwrite=False,
):
    """
    Auto-annotate images using OpenCV and save YOLO-format .txt label files.

    Args:
        images      : List of image paths (.tif / .jpg / .png etc.)
        output_dir  : Directory where .txt label files are saved.
        class_id    : YOLO class index to write for every detection.
                      0 = PbI2 excess (bright particles)
                      1 = pinholes    (dark holes)
                      2 = 3D-2D mixed pinholes
        method      : Detection method — choose based on what you are detecting:
                      ┌─────────────────────┬────────────────────────────────────────────────────┐
                      │ "threshold"         │ Dark features (pinholes) — global dark threshold   │
                      │ "threshold_bright"  │ Bright features (PbI2) — global bright threshold   │
                      │                     │   Works well when particles are clearly white       │
                      │ "tophat_bright"     │ Small/faint bright features (PbI2 in 3D-2D images) │
                      │                     │   Best when particles have LOW contrast vs grains   │
                      │                     │   Uses morphological top-hat — ignores background   │
                      │ "color_mask"        │ Dark features via HSV value channel                │
                      │ "color_mask_bright" │ Bright features via HSV value channel              │
                      │ "adaptive"          │ Dark features relative to local neighbourhood      │
                      │ "canny"             │ Edges — general purpose                            │
                      │ "watershed"         │ Advanced dark region segmentation                  │
                      └─────────────────────┴────────────────────────────────────────────────────┘
        threshold1  : Main threshold value:
                      - "threshold" / "threshold_bright": pixel intensity cutoff (0–255)
                      - "tophat_bright": sensitivity after top-hat (lower = more detections)
                      - "color_mask" / "color_mask_bright": HSV value cutoff
                      - "canny": lower edge threshold
        threshold2  : Upper threshold for Canny only.
        brightness  : Pixel brightness offset (-100 … 100). Keep 0 for PbI2.
        contrast    : Contrast multiplier (0.5 … 3.0).
        min_area    : Minimum contour area in px² — filters noise.
        max_area    : Maximum contour area in px² — filters large background regions.
        use_clahe   : Apply CLAHE contrast enhancement before detection.
        clahe_clip  : CLAHE clip limit.
        clahe_grid  : CLAHE tile grid size.
        overwrite   : If True, overwrite existing label files.

    Returns:
        Number of images for which a label file was written (including empty ones).

    ── Recommended settings per class ───────────────────────────────────────────
    PbI2 excess — large/bright particles (previous sample):
        method="threshold_bright", threshold1=160, contrast=1.4,
        brightness=0, min_area=30, max_area=4000, class_id=0

    PbI2 excess — small/faint particles (3D-2D morphology, this sample):
        method="tophat_bright", threshold1=15, contrast=1.0,
        brightness=0, min_area=10, max_area=2000, class_id=0

    Pinholes (dark holes):
        method="threshold", threshold1=80, contrast=1.2,
        brightness=0, min_area=50, max_area=10000, class_id=1
    """
    os.makedirs(output_dir, exist_ok=True)
    labeled_count = 0

    for img_path in images:
        base_name  = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(output_dir, base_name + ".txt")

        if os.path.exists(label_path) and not overwrite:
            continue

        # Use PIL to load — handles .tif 16-bit / RGBA that cv2.imread silently drops.
        try:
            with Image.open(img_path) as pil_img:
                rgb  = np.array(pil_img.convert("RGB"))
                img  = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        except Exception as e:
            print(f"ERROR loading {img_path}: {e}")
            continue

        h, w = gray.shape

        # ── Optional CLAHE ────────────────────────────────────────────────────
        if use_clahe:
            clahe = cv2.createCLAHE(
                clipLimit=clahe_clip,
                tileGridSize=(clahe_grid, clahe_grid)
            )
            gray = clahe.apply(gray)

        # ── Brightness / contrast ─────────────────────────────────────────────
        adjusted = cv2.convertScaleAbs(gray, alpha=contrast, beta=brightness)

        # ── Detection method ──────────────────────────────────────────────────

        if method == "threshold":
            # Dark features (pinholes, voids).
            _, binary = cv2.threshold(adjusted, threshold1, 255, cv2.THRESH_BINARY_INV)
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=2)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        elif method == "threshold_bright":
            # Bright features — global threshold.
            # Works when particles are clearly whiter than all grain surfaces.
            _, binary = cv2.threshold(adjusted, threshold1, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)

        elif method == "tophat_bright":
            # ── TOP-HAT TRANSFORM (new method for small/faint PbI2 particles) ──
            #
            # Problem this solves:
            #   In 3D-2D morphology images, PbI2 particles are small and their
            #   brightness is similar to grain edge highlights. A global threshold
            #   either misses the particles (threshold too high) or floods the
            #   image with grain edges (threshold too low).
            #
            # How top-hat works:
            #   top_hat = image − morphological_opening(image, kernel)
            #   Opening with a LARGE kernel removes bright structures SMALLER
            #   than the kernel (i.e. particles) but keeps the slowly-varying
            #   grain background. Subtracting gives an image where ONLY the
            #   small bright particles remain — grain background becomes ~0.
            #   Then a low threshold on that residual catches all particles
            #   without touching grain edges at all.
            #
            # threshold1 controls sensitivity:
            #   Lower  (10–20) = more detections, may include tiny noise specks
            #   Higher (25–40) = only the brightest particles
            #   Start with threshold1=15, tune min_area to filter noise.
            #
            # kernel_size controls what "small" means:
            #   Should be larger than any particle but smaller than a grain.
            #   21px works well for 1024px SEM images of this magnification.

            kernel_size = 21    # px — larger than particles, smaller than grains
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )

            # Top-hat: keeps only bright features smaller than kernel_size.
            tophat = cv2.morphologyEx(adjusted, cv2.MORPH_TOPHAT, kernel)

            # Threshold the top-hat residual image — now even faint particles
            # stand clearly above the near-zero background.
            _, binary = cv2.threshold(tophat, threshold1, 255, cv2.THRESH_BINARY)

            # Small closing to merge fragmented detections of the same particle.
            close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k, iterations=1)

        elif method == "adaptive":
            # Dark features relative to local neighbourhood.
            binary = cv2.adaptiveThreshold(
                adjusted, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=2)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        elif method == "color_mask":
            # Dark features via HSV value channel (pinholes).
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower_dark  = np.array([0,   0,   0])
            upper_dark  = np.array([180, 255, threshold1])
            mask_dark   = cv2.inRange(hsv, lower_dark, upper_dark)
            lower_brown = np.array([0,  30, 0])
            upper_brown = np.array([20, 255, threshold1])
            mask_brown  = cv2.inRange(hsv, lower_brown, upper_brown)
            binary = cv2.bitwise_or(mask_dark, mask_brown)
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=2)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        elif method == "color_mask_bright":
            # Bright features via HSV value channel.
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower_bright = np.array([0,   0,   threshold1])
            upper_bright = np.array([180, 60,  255])
            binary = cv2.inRange(hsv, lower_bright, upper_bright)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)

        elif method == "canny":
            edges  = cv2.Canny(adjusted, threshold1, threshold2)
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.dilate(edges, kernel, iterations=1)

        elif method == "watershed":
            _, binary_inv = cv2.threshold(adjusted, threshold1, 255, cv2.THRESH_BINARY_INV)
            kernel   = np.ones((3, 3), np.uint8)
            opening  = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel, iterations=2)
            sure_bg  = cv2.dilate(opening, kernel, iterations=3)
            dist     = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist, 0.3 * dist.max(), 255, 0)
            sure_fg  = np.uint8(sure_fg)
            unknown  = cv2.subtract(sure_bg, sure_fg)
            _, markers = cv2.connectedComponents(sure_fg)
            markers  = markers + 1
            markers[unknown == 255] = 0
            markers  = cv2.watershed(cv2.cvtColor(adjusted, cv2.COLOR_GRAY2BGR), markers)
            binary   = np.uint8(markers == -1) * 255

        else:
            binary = adjusted

        # ── Find contours → YOLO boxes ────────────────────────────────────────
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        yolo_lines = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            width    = bw / w
            height   = bh / h

            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        # Always write — empty file marks image as processed (no re-run on next call).
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))

        labeled_count += 1

    return labeled_count