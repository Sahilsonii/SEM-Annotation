import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


DCT_BLOCK_SIZE   = 30
CLAHE_CLIP       = 2.0
CLAHE_GRID       = 8
KERNEL_SIZE      = 15
MIN_AREA         = 80
MAX_AREA         = 8000
MIN_CIRCULARITY  = 0.35
CONFIDENCE_THRESH = 0.45
OUTPUT_PATH      = "dct_pinhole_result.png"


def load_grayscale(image_path):
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return img_gray, img_color


def apply_dct_filter(gray, block_size=DCT_BLOCK_SIZE):
    f32 = np.float32(gray)
    dct = cv2.dct(f32)
    mask = np.ones_like(dct)
    mask[:block_size, :block_size] = 0
    filtered_dct = dct * mask
    reconstructed = cv2.idct(filtered_dct)
    reconstructed = np.clip(reconstructed, 0, 255)
    return np.uint8(reconstructed)


def apply_clahe(gray, clip=CLAHE_CLIP, grid=CLAHE_GRID):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return clahe.apply(gray)


def apply_black_hat(gray, ksize=KERNEL_SIZE):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)


def threshold_otsu(gray):
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def morphological_open(binary, ksize=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)


def detect_contours(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def filter_contours(contours, min_area=MIN_AREA, max_area=MAX_AREA, min_circ=MIN_CIRCULARITY):
    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < min_circ:
            continue
        valid.append((c, area, circularity))
    return valid


def compute_confidence(gray_original, contour, area):
    x, y, w, h = cv2.boundingRect(contour)
    roi = gray_original[y:y+h, x:x+w]
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = contour - contour.min(axis=0)[0]
    cv2.drawContours(mask, [shifted], -1, 255, -1)

    interior_pixels = roi[mask == 255]
    exterior_pixels = roi[mask == 0]

    if len(interior_pixels) == 0 or len(exterior_pixels) == 0:
        return 0.0

    interior_mean = np.mean(interior_pixels)
    exterior_mean = np.mean(exterior_pixels)
    contrast      = (exterior_mean - interior_mean) / (exterior_mean + 1e-6)
    contrast      = np.clip(contrast, 0, 1)

    size_score = np.clip(area / 1000.0, 0, 1)
    confidence = 0.75 * contrast + 0.25 * size_score
    return float(confidence)


def draw_detections(img_color, detections):
    out = img_color.copy()
    for (x, y, w, h, conf) in detections:
        color = (0, 255, 0) if conf > 0.65 else (0, 200, 255)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        label = f"pinhole {conf:.2f}"
        cv2.putText(out, label, (x, max(y - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def run_pipeline(image_path, output_path=OUTPUT_PATH):
    gray, color = load_grayscale(image_path)

    dct_filtered    = apply_dct_filter(gray)
    clahe_img       = apply_clahe(dct_filtered)
    black_hat       = apply_black_hat(clahe_img)
    binary_otsu     = threshold_otsu(black_hat)
    binary_clean    = morphological_open(binary_otsu)
    contours        = detect_contours(binary_clean)
    valid_contours  = filter_contours(contours)

    detections = []
    for (c, area, _circularity) in valid_contours:
        conf = compute_confidence(gray, c, area)
        if conf >= CONFIDENCE_THRESH:
            x, y, w, h = cv2.boundingRect(c)
            detections.append((x, y, w, h, conf))

    result_img = draw_detections(color, detections)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(dct_filtered,  cmap="gray");  axes[0].set_title("DCT Filtered")
    axes[1].imshow(clahe_img,     cmap="gray");  axes[1].set_title("CLAHE")
    axes[2].imshow(black_hat,     cmap="gray");  axes[2].set_title("Black Hat")
    axes[3].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    axes[3].set_title(f"Final Detections ({len(detections)})")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Detections: {len(detections)}  |  Saved: {output_path}")
    return detections


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dct_pinhole_detect.py <path/to/image>")
        sys.exit(1)
    run_pipeline(sys.argv[1])
