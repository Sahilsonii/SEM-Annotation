"""
PerovskiteDefectFilter (PDF) v1.0
=================================
An opencv_contrib-ready adaptive defect detection filter for perovskite
solar cell SEM images. Designed for single-line usage:

    filter = PerovskiteDefectFilter(mode="auto")
    detections = filter.detect(image)

Modes:
    "pbi2"     — Bright particle detection (PbI2 excess crystals/needles)
    "pinhole"  — Dark pit/hole detection (small + large pinholes)
    "3d"       — 3D perovskite: grain boundary suppression + both paths
    "3d_2d"    — 3D-2D mixed: handles needle crystals + pinholes + grains
    "2d"       — 2D perovskite: flatter morphology, subtle defects
    "auto"     — Automatic mode selection from image statistics

Zero deep learning. Pure classical image processing.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict
from enum import Enum


class DefectType(Enum):
    PBI2_BRIGHT = 0
    PINHOLE_SMALL = 1
    PINHOLE_LARGE = 2
    PBI2_NEEDLE = 3
    GRAIN_BOUNDARY_DEFECT = 4


class FilterMode(Enum):
    AUTO = "auto"
    PBI2 = "pbi2"
    PINHOLE = "pinhole"
    MODE_3D = "3d"
    MODE_3D_2D = "3d_2d"
    MODE_2D = "2d"


@dataclass
class Defect:
    x: int
    y: int
    w: int
    h: int
    class_id: int
    defect_type: str
    confidence: float
    area: float
    circularity: float
    solidity: float
    contrast: float
    interior_mean: float
    source: str

    def to_yolo(self, img_w: int, img_h: int) -> str:
        xc = (self.x + self.w / 2.0) / img_w
        yc = (self.y + self.h / 2.0) / img_h
        wn = self.w / img_w
        hn = self.h / img_h
        return f"{self.class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _ModeProfile:
    clahe_clip: float = 3.0
    clahe_grid: int = 8
    gaussian_sigma: float = 1.0
    tophat_kernels: list = field(default_factory=lambda: [3, 5, 7])
    blackhat_kernels: list = field(default_factory=lambda: [5, 7, 11])
    bright_percentile_hi: float = 95.0
    bright_percentile_lo: float = 88.0
    dark_percentile: float = 12.0
    dark_percentile_strict: float = 6.0
    min_area_bright: int = 4
    max_area_bright: int = 2000
    min_area_dark: int = 15
    max_area_dark: int = 20000
    min_circularity_bright: float = 0.20
    min_circularity_dark: float = 0.30
    min_solidity: float = 0.40
    max_aspect_ratio: float = 5.0
    max_fill_ratio: float = 0.08
    conf_thresh: float = 0.30
    nms_iou: float = 0.40
    suppress_grain_boundaries: bool = False
    grain_boundary_canny_lo: int = 30
    grain_boundary_canny_hi: int = 80
    grain_boundary_dilate: int = 5
    detect_bright: bool = True
    detect_dark: bool = True
    detect_needles: bool = False
    needle_min_aspect: float = 2.5
    needle_min_area: int = 10


_PROFILES = {
    FilterMode.PBI2: _ModeProfile(
        tophat_kernels=[3, 5, 7],
        detect_bright=True,
        detect_dark=False,
        detect_needles=True,
        min_area_bright=3,
        max_area_bright=3000,
        bright_percentile_hi=96.0,
        bright_percentile_lo=90.0,
        min_circularity_bright=0.15,
        suppress_grain_boundaries=False,
    ),
    FilterMode.PINHOLE: _ModeProfile(
        blackhat_kernels=[5, 7, 11, 15],
        detect_bright=False,
        detect_dark=True,
        min_area_dark=4,
        dark_percentile=12.0,
        dark_percentile_strict=6.0,
        min_circularity_dark=0.25,
        suppress_grain_boundaries=False,
    ),
    FilterMode.MODE_3D: _ModeProfile(
        tophat_kernels=[3, 5],
        blackhat_kernels=[5, 7, 11],
        detect_bright=True,
        detect_dark=True,
        detect_needles=True,
        suppress_grain_boundaries=True,
        grain_boundary_dilate=7,
        min_area_bright=3,
        max_area_bright=1500,
        min_area_dark=10,
        bright_percentile_hi=95.0,
        bright_percentile_lo=88.0,
        dark_percentile=10.0,
        min_circularity_bright=0.15,
        min_circularity_dark=0.30,
        max_fill_ratio=0.06,
        conf_thresh=0.25,
    ),
    FilterMode.MODE_3D_2D: _ModeProfile(
        tophat_kernels=[3, 5, 7],
        blackhat_kernels=[5, 7, 11],
        detect_bright=True,
        detect_dark=True,
        detect_needles=True,
        suppress_grain_boundaries=True,
        grain_boundary_dilate=5,
        min_area_bright=3,
        max_area_bright=2000,
        min_area_dark=8,
        bright_percentile_hi=94.0,
        bright_percentile_lo=87.0,
        dark_percentile=10.0,
        min_circularity_bright=0.12,
        min_circularity_dark=0.25,
        needle_min_aspect=2.0,
        needle_min_area=8,
        conf_thresh=0.25,
    ),
    FilterMode.MODE_2D: _ModeProfile(
        tophat_kernels=[3, 5, 7, 9],
        blackhat_kernels=[7, 11, 15],
        detecti _bright=True,
        detect_dark=True,
        detect_needles=False,
        suppress_grain_boundaries=False,
        min_area_bright=5,
        bright_percentile_hi=95.0,
        dark_percentile=15.0,
        min_circularity_bright=0.20,
        conf_thresh=0.30,
    ),
}


class PerovskiteDefectFilter:
    """
    Single-line defect detection for perovskite SEM images.

    Usage:
        pdf = PerovskiteDefectFilter(mode="auto")
        detections = pdf.detect(image)
        pdf.save_yolo(detections, image.shape, "output.txt")
    """

    VERSION = "1.0.0"

    def __init__(self, mode: str = "auto"):
        if isinstance(mode, str):
            mode = FilterMode(mode.lower())
        self._mode = mode
        self._profile = None if mode == FilterMode.AUTO else _PROFILES[mode]
        self._debug_stages = {}
        self._stats = {}

    @classmethod
    def create_pbi2_detector(cls):
        return cls(mode="pbi2")

    @classmethod
    def create_pinhole_detector(cls):
        return cls(mode="pinhole")

    @classmethod
    def create_3d_detector(cls):
        return cls(mode="3d")

    @classmethod
    def create_3d_2d_detector(cls):
        return cls(mode="3d_2d")

    @classmethod
    def create_2d_detector(cls):
        return cls(mode="2d")

    # ── PUBLIC API ────────────────────────────────────────────────────────

    def detect(self, image: np.ndarray) -> List[Defect]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        gray, crop_y = self._crop_sem_bar(gray)
        self._debug_stages["01_cropped"] = gray.copy()

        if self._mode == FilterMode.AUTO:
            detected_mode = self._auto_detect_mode(gray)
            self._profile = _PROFILES[detected_mode]
            self._stats["auto_mode"] = detected_mode.value
        else:
            detected_mode = self._mode

        p = self._profile
        enhanced = self._preprocess(gray, p)
        self._debug_stages["02_enhanced"] = enhanced.copy()

        grain_mask = None
        if p.suppress_grain_boundaries:
            grain_mask = self._build_grain_boundary_mask(enhanced, p)
            self._debug_stages["03_grain_boundary_mask"] = grain_mask.copy()

        all_detections = []

        if p.detect_bright:
            bright_mask = self._detect_bright_particles(enhanced, p)
            if grain_mask is not None:
                bright_mask = cv2.bitwise_and(bright_mask, cv2.bitwise_not(grain_mask))
            self._debug_stages["04_bright_mask"] = bright_mask.copy()
            bright_dets = self._extract_bright_defects(bright_mask, gray, p)
            all_detections.extend(bright_dets)

        if p.detect_dark:
            dark_mask = self._detect_dark_pits(enhanced, p)
            if grain_mask is not None:
                dark_mask = cv2.bitwise_and(dark_mask, cv2.bitwise_not(grain_mask))
            self._debug_stages["05_dark_mask"] = dark_mask.copy()
            dark_dets = self._extract_dark_defects(dark_mask, gray, p)
            all_detections.extend(dark_dets)

        if p.detect_needles:
            needle_mask = self._detect_needle_crystals(enhanced, gray, p)
            if grain_mask is not None:
                needle_mask = cv2.bitwise_and(needle_mask, cv2.bitwise_not(grain_mask))
            self._debug_stages["06_needle_mask"] = needle_mask.copy()
            needle_dets = self._extract_needle_defects(needle_mask, gray, p)
            all_detections.extend(needle_dets)

        detections = self._nms(all_detections, p.nms_iou)

        self._stats["total_detections"] = len(detections)
        self._stats["mode"] = detected_mode.value
        type_counts = {}
        for d in detections:
            type_counts[d.defect_type] = type_counts.get(d.defect_type, 0) + 1
        self._stats["defect_counts"] = type_counts

        return detections

    def get_debug_stages(self) -> Dict[str, np.ndarray]:
        return self._debug_stages.copy()

    def get_stats(self) -> dict:
        return self._stats.copy()

    @staticmethod
    def save_yolo(detections: List[Defect], img_shape: Tuple[int, ...],
                  path: str):
        h, w = img_shape[:2]
        lines = [d.to_yolo(w, h) for d in detections]
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write("\n".join(lines))

    @staticmethod
    def annotate(image: np.ndarray, detections: List[Defect]) -> np.ndarray:
        out = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        colors = {
            "pbi2_bright": (0, 255, 255),
            "pbi2_needle": (0, 200, 255),
            "pinhole_small": (0, 255, 0),
            "pinhole_large": (255, 100, 100),
            "grain_boundary_defect": (255, 0, 255),
        }
        for d in detections:
            c = colors.get(d.defect_type, (255, 255, 255))
            t = 1 if d.area < 100 else 2
            cv2.rectangle(out, (d.x, d.y), (d.x + d.w, d.y + d.h), c, t)
            label = f"{d.defect_type[:6]} {d.confidence:.2f}"
            fs = 0.30 if d.area < 100 else 0.40
            cv2.putText(out, label, (d.x, max(d.y - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, c, 1, cv2.LINE_AA)
        return out

    def visualize_pipeline(self, image: np.ndarray, detections: List[Defect],
                           output_path: str):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        stages = self._debug_stages
        annotated = self.annotate(image, detections)

        display = {}
        if "02_enhanced" in stages:
            display["CLAHE Enhanced"] = stages["02_enhanced"]
        if "03_grain_boundary_mask" in stages:
            display["Grain Boundary\nSuppression"] = stages["03_grain_boundary_mask"]
        if "04_bright_mask" in stages:
            display["Bright Particle\nMask (PbI₂)"] = stages["04_bright_mask"]
        if "05_dark_mask" in stages:
            display["Dark Pit Mask\n(Pinholes)"] = stages["05_dark_mask"]
        if "06_needle_mask" in stages:
            display["Needle Crystal\nMask"] = stages["06_needle_mask"]
        display[f"Detections ({len(detections)})"] = annotated

        n = len(display)
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
        if n == 1:
            axes = [axes]
        for ax, (title, img) in zip(axes, display.items()):
            if len(img.shape) == 2:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.axis("off")
        plt.suptitle(f"PerovskiteDefectFilter v{self.VERSION} | Mode: {self._stats.get('mode', '?')}",
                     fontsize=11, fontweight="bold", y=1.02)
        plt.tight_layout()
        import os
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── SEM BAR REMOVAL ───────────────────────────────────────────────────

    def _crop_sem_bar(self, gray: np.ndarray) -> Tuple[np.ndarray, int]:
        h, w = gray.shape
        scan = gray[int(h * 0.82):, :]
        row_medians = np.median(scan, axis=1)
        dark_rows = np.where(row_medians < 20)[0]
        if len(dark_rows) > 0:
            cut_y = int(h * 0.82) + dark_rows[0] - 5
            cut_y = max(int(h * 0.70), cut_y)
        else:
            col_std = np.std(scan, axis=1)
            uniform_rows = np.where(col_std < 5)[0]
            if len(uniform_rows) > 3:
                cut_y = int(h * 0.82) + uniform_rows[0] - 3
                cut_y = max(int(h * 0.70), cut_y)
            else:
                cut_y = h
        return gray[:cut_y, :], cut_y

    # ── AUTO MODE DETECTION ───────────────────────────────────────────────

    def _auto_detect_mode(self, gray: np.ndarray) -> FilterMode:
        h, w = gray.shape
        mean_i = float(np.mean(gray))
        std_i = float(np.std(gray))
        cv_coeff = std_i / (mean_i + 1e-6)

        bright_pct = float(np.percentile(gray, 97))
        dark_pct = float(np.percentile(gray, 3))
        dynamic_range = bright_pct - dark_pct

        edges = cv2.Canny(gray, 30, 80)
        edge_density = float(np.sum(edges > 0)) / (h * w)

        gx = cv2.Sobel(gray.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        grad_mean = float(np.mean(grad_mag))

        tophat_3 = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        bright_spot_density = float(np.sum(tophat_3 > 40)) / (h * w)

        self._stats["auto_detect"] = {
            "mean": round(mean_i, 1), "std": round(std_i, 1),
            "cv": round(cv_coeff, 3), "dynamic_range": round(dynamic_range, 1),
            "edge_density": round(edge_density, 4),
            "grad_mean": round(grad_mean, 2),
            "bright_spot_density": round(bright_spot_density, 5),
        }

        if edge_density > 0.08 and grad_mean > 20:
            if bright_spot_density > 0.005:
                return FilterMode.MODE_3D
            else:
                return FilterMode.MODE_3D
        elif edge_density > 0.04:
            if bright_spot_density > 0.003:
                return FilterMode.MODE_3D_2D
            else:
                return FilterMode.MODE_3D_2D
        else:
            if bright_spot_density > 0.008:
                return FilterMode.PBI2
            else:
                return FilterMode.MODE_2D

    # ── PREPROCESSING ─────────────────────────────────────────────────────

    def _preprocess(self, gray: np.ndarray, p: _ModeProfile) -> np.ndarray:
        ksize = max(3, int(np.ceil(p.gaussian_sigma * 6)) | 1)
        denoised = cv2.GaussianBlur(gray, (ksize, ksize), p.gaussian_sigma)
        clahe = cv2.createCLAHE(clipLimit=p.clahe_clip,
                                 tileGridSize=(p.clahe_grid, p.clahe_grid))
        return clahe.apply(denoised)

    # ── GRAIN BOUNDARY SUPPRESSION ────────────────────────────────────────

    def _build_grain_boundary_mask(self, enhanced: np.ndarray,
                                    p: _ModeProfile) -> np.ndarray:
        edges = cv2.Canny(enhanced, p.grain_boundary_canny_lo,
                          p.grain_boundary_canny_hi)

        k_dilate = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (p.grain_boundary_dilate, p.grain_boundary_dilate)
        )
        dilated = cv2.dilate(edges, k_dilate, iterations=1)

        gx = cv2.Sobel(enhanced.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(enhanced.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)

        grad_thresh = np.percentile(grad_mag, 80)
        high_grad = (grad_mag > grad_thresh).astype(np.uint8) * 255

        k_thin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        high_grad = cv2.morphologyEx(high_grad, cv2.MORPH_CLOSE, k_thin)

        boundary_mask = cv2.bitwise_or(dilated, high_grad)

        k_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        boundary_mask = cv2.morphologyEx(boundary_mask, cv2.MORPH_CLOSE, k_smooth)

        return boundary_mask

    # ── BRIGHT PARTICLE DETECTION (PbI₂ excess) ──────────────────────────

    def _detect_bright_particles(self, enhanced: np.ndarray,
                                  p: _ModeProfile) -> np.ndarray:
        combined = np.zeros_like(enhanced, dtype=np.float64)

        for ks in p.tophat_kernels:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
            tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
            combined = np.maximum(combined, tophat.astype(np.float64))

        combined_u8 = np.clip(combined, 0, 255).astype(np.uint8)

        hi_val = np.percentile(combined_u8[combined_u8 > 0], p.bright_percentile_hi) \
            if np.any(combined_u8 > 0) else 128
        lo_val = np.percentile(combined_u8[combined_u8 > 0], p.bright_percentile_lo) \
            if np.any(combined_u8 > 0) else 64

        _, mask_hi = cv2.threshold(combined_u8, int(max(hi_val, 5)), 255, cv2.THRESH_BINARY)

        _, mask_lo = cv2.threshold(combined_u8, int(max(lo_val, 3)), 255, cv2.THRESH_BINARY)

        bright_pct = np.percentile(enhanced, 92)
        _, raw_bright = cv2.threshold(enhanced, int(bright_pct), 255, cv2.THRESH_BINARY)

        merged = cv2.bitwise_or(mask_hi, cv2.bitwise_and(mask_lo, raw_bright))

        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, k_open, iterations=1)

        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, k_close, iterations=1)

        total_px = merged.shape[0] * merged.shape[1]
        fill = cv2.countNonZero(merged) / total_px
        if fill > p.max_fill_ratio:
            _, merged = cv2.threshold(combined_u8, int(max(hi_val * 1.2, 10)),
                                       255, cv2.THRESH_BINARY)
            merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, k_open)

        return merged

    # ── DARK PIT DETECTION (Pinholes) ─────────────────────────────────────

    def _detect_dark_pits(self, enhanced: np.ndarray,
                           p: _ModeProfile) -> np.ndarray:
        dark_pct = float(np.percentile(enhanced, p.dark_percentile))
        _, binary = cv2.threshold(enhanced, int(max(dark_pct, 3)), 255,
                                   cv2.THRESH_BINARY_INV)

        total_px = binary.shape[0] * binary.shape[1]
        fill = cv2.countNonZero(binary) / total_px
        if fill > p.max_fill_ratio:
            dark_pct = float(np.percentile(enhanced, p.dark_percentile_strict))
            _, binary = cv2.threshold(enhanced, int(max(dark_pct, 3)), 255,
                                       cv2.THRESH_BINARY_INV)

        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open, iterations=1)

        micro_pct = float(np.percentile(enhanced, 3))
        _, micro = cv2.threshold(enhanced, int(max(micro_pct, 3)), 255,
                                  cv2.THRESH_BINARY_INV)
        k_micro = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        micro = cv2.morphologyEx(micro, cv2.MORPH_CLOSE, k_micro, iterations=1)

        combined = cv2.bitwise_or(binary, micro)
        return combined

    # ── NEEDLE CRYSTAL DETECTION ──────────────────────────────────────────

    def _detect_needle_crystals(self, enhanced: np.ndarray,
                                 gray: np.ndarray,
                                 p: _ModeProfile) -> np.ndarray:
        tophat_large = cv2.morphologyEx(
            enhanced, cv2.MORPH_TOPHAT,
            cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        )

        bright_thresh = np.percentile(tophat_large, 92)
        _, mask = cv2.threshold(tophat_large, int(max(bright_thresh, 5)),
                                 255, cv2.THRESH_BINARY)

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

        return mask

    # ── CONTOUR EXTRACTION ────────────────────────────────────────────────

    def _compute_contour_features(self, contour, gray: np.ndarray):
        area = cv2.contourArea(contour)
        perim = cv2.arcLength(contour, True)
        circ = 4.0 * np.pi * area / (perim ** 2) if perim > 1 else 0
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-6)
        x, y, w, h = cv2.boundingRect(contour)
        aspect = max(w, h) / (min(w, h) + 1e-6)

        mask_full = np.zeros(gray.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask_full, [contour], -1, 255, -1)
        interior = gray[mask_full == 255]
        interior_mean = float(np.mean(interior)) if len(interior) > 0 else 0

        pad = max(w, h, 12)
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(gray.shape[1], x + w + pad)
        y1 = min(gray.shape[0], y + h + pad)
        roi = gray[y0:y1, x0:x1]
        roi_mask = mask_full[y0:y1, x0:x1]
        ext_pixels = roi[roi_mask == 0]
        ext_mean = float(np.mean(ext_pixels)) if len(ext_pixels) > 0 else 128

        if ext_mean > 1:
            contrast = abs(ext_mean - interior_mean) / ext_mean
        else:
            contrast = 0.0
        contrast = float(np.clip(contrast, 0, 1))

        return {
            "area": area, "perimeter": perim, "circularity": circ,
            "solidity": solidity, "x": x, "y": y, "w": w, "h": h,
            "aspect": aspect, "interior_mean": interior_mean,
            "exterior_mean": ext_mean, "contrast": contrast,
        }

    def _extract_bright_defects(self, mask: np.ndarray, gray: np.ndarray,
                                 p: _ModeProfile) -> List[Defect]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        img_median = float(np.median(gray))

        for c in contours:
            f = self._compute_contour_features(c, gray)
            if f["area"] < p.min_area_bright or f["area"] > p.max_area_bright:
                continue
            if f["circularity"] < p.min_circularity_bright:
                continue
            if f["solidity"] < p.min_solidity:
                continue
            if f["aspect"] > p.max_aspect_ratio:
                continue
            if f["interior_mean"] < img_median:
                continue
            if f["contrast"] < 0.08:
                continue

            area_norm = float(np.clip(f["area"] / 500.0, 0, 1))
            conf = (0.40 * f["contrast"] +
                    0.25 * f["circularity"] +
                    0.15 * f["solidity"] +
                    0.10 * area_norm + 0.10)
            conf = float(np.clip(conf, 0, 1))

            if conf < p.conf_thresh:
                continue

            dets.append(Defect(
                x=f["x"], y=f["y"], w=f["w"], h=f["h"],
                class_id=DefectType.PBI2_BRIGHT.value,
                defect_type="pbi2_bright",
                confidence=conf, area=f["area"],
                circularity=f["circularity"], solidity=f["solidity"],
                contrast=f["contrast"], interior_mean=f["interior_mean"],
                source="bright",
            ))
        return dets

    def _extract_dark_defects(self, mask: np.ndarray, gray: np.ndarray,
                               p: _ModeProfile) -> List[Defect]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        img_median = float(np.median(gray))
        img_std = float(np.std(gray))
        dark_gate = img_median - 0.75 * img_std

        for c in contours:
            f = self._compute_contour_features(c, gray)
            if f["area"] < p.min_area_dark or f["area"] > p.max_area_dark:
                continue

            dynamic_circ = p.min_circularity_dark
            if f["area"] > 500:
                dynamic_circ = max(0.15, dynamic_circ - 0.20)
            elif f["area"] > 200:
                dynamic_circ = max(0.20, dynamic_circ - 0.10)

            if f["circularity"] < dynamic_circ:
                continue
            if f["solidity"] < p.min_solidity:
                continue
            if f["aspect"] > p.max_aspect_ratio:
                continue
            if f["interior_mean"] > dark_gate:
                continue
            if f["contrast"] < 0.10:
                continue

            is_large = f["area"] > 300
            area_norm = float(np.clip(f["area"] / 2000.0, 0, 1))
            conf = (0.40 * f["contrast"] +
                    0.25 * f["circularity"] +
                    0.15 * f["solidity"] +
                    0.10 * area_norm + 0.10)
            conf = float(np.clip(conf, 0, 1))

            if conf < p.conf_thresh:
                continue

            dtype = DefectType.PINHOLE_LARGE if is_large else DefectType.PINHOLE_SMALL
            dets.append(Defect(
                x=f["x"], y=f["y"], w=f["w"], h=f["h"],
                class_id=dtype.value,
                defect_type="pinhole_large" if is_large else "pinhole_small",
                confidence=conf, area=f["area"],
                circularity=f["circularity"], solidity=f["solidity"],
                contrast=f["contrast"], interior_mean=f["interior_mean"],
                source="dark",
            ))
        return dets

    def _extract_needle_defects(self, mask: np.ndarray, gray: np.ndarray,
                                 p: _ModeProfile) -> List[Defect]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        img_median = float(np.median(gray))

        for c in contours:
            f = self._compute_contour_features(c, gray)
            if f["area"] < p.needle_min_area:
                continue
            if f["area"] > p.max_area_bright:
                continue
            if f["aspect"] < p.needle_min_aspect:
                continue
            if f["interior_mean"] < img_median:
                continue
            if f["contrast"] < 0.06:
                continue

            conf = (0.35 * f["contrast"] +
                    0.20 * min(f["aspect"] / 5.0, 1.0) +
                    0.20 * f["solidity"] +
                    0.15 * float(np.clip(f["area"] / 300.0, 0, 1)) + 0.10)
            conf = float(np.clip(conf, 0, 1))

            if conf < p.conf_thresh:
                continue

            dets.append(Defect(
                x=f["x"], y=f["y"], w=f["w"], h=f["h"],
                class_id=DefectType.PBI2_NEEDLE.value,
                defect_type="pbi2_needle",
                confidence=conf, area=f["area"],
                circularity=f["circularity"], solidity=f["solidity"],
                contrast=f["contrast"], interior_mean=f["interior_mean"],
                source="needle",
            ))
        return dets

    # ── NMS ───────────────────────────────────────────────────────────────

    @staticmethod
    def _nms(detections: List[Defect],
             iou_thresh: float = 0.4) -> List[Defect]:
        if not detections:
            return []
        dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        keep = []
        for d in dets:
            b1 = (d.x, d.y, d.w, d.h)
            if all(_iou(b1, (k.x, k.y, k.w, k.h)) < iou_thresh for k in keep):
                keep.append(d)
        return keep


def _iou(a, b):
    xa = max(a[0], b[0])
    ya = max(a[1], b[1])
    xb = min(a[0] + a[2], b[0] + b[2])
    yb = min(a[1] + a[3], b[1] + b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / (union + 1e-6)
