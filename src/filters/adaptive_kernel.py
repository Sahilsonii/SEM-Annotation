import cv2
import numpy as np


class AdaptiveKernelEngine:

    KERNEL_SIZES = np.array([3, 5, 7, 11, 15, 21])

    def __init__(self, variance_weight=0.4, gradient_weight=0.35, entropy_weight=0.25):
        self.variance_weight = variance_weight
        self.gradient_weight = gradient_weight
        self.entropy_weight = entropy_weight
        self._kernel_stats = {}

    def compute_local_variance_map(self, gray, window=21):
        gray_f = gray.astype(np.float64)
        mean = cv2.blur(gray_f, (window, window))
        sqr_mean = cv2.blur(gray_f ** 2, (window, window))
        variance = sqr_mean - mean ** 2
        variance = np.clip(variance, 0, None)
        return variance

    def compute_gradient_magnitude_map(self, gray):
        gray_f = gray.astype(np.float64)
        gx = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        return mag

    def compute_fast_entropy_map(self, gray, window=15):
        gray_q = (gray.astype(np.float64) / 16.0).astype(np.uint8)
        gray_q = np.clip(gray_q, 0, 15)
        h, w = gray.shape
        entropy_map = np.zeros((h, w), dtype=np.float64)
        n_bins = 16
        pad = window // 2
        padded = np.pad(gray_q, pad, mode="reflect")
        area = window * window
        one_hot = np.zeros((h + 2 * pad, w + 2 * pad, n_bins), dtype=np.float32)
        for b in range(n_bins):
            one_hot[:, :, b] = (padded == b).astype(np.float32)
        for b in range(n_bins):
            one_hot[:, :, b] = cv2.blur(one_hot[:, :, b], (window, window))
        prob_map = one_hot[pad:pad + h, pad:pad + w, :]
        prob_map = np.clip(prob_map, 1e-10, 1.0)
        entropy_map = -np.sum(prob_map * np.log2(prob_map), axis=2)
        return entropy_map

    def compute_kernel_size_map(self, variance_map, gradient_map, entropy_map):
        def _normalize(m):
            mn, mx = m.min(), m.max()
            if mx - mn < 1e-8:
                return np.zeros_like(m)
            return (m - mn) / (mx - mn)

        v_norm = _normalize(variance_map)
        g_norm = _normalize(gradient_map)
        e_norm = _normalize(entropy_map)

        complexity = (
            self.variance_weight * v_norm +
            self.gradient_weight * g_norm +
            self.entropy_weight * e_norm
        )

        n_sizes = len(self.KERNEL_SIZES)
        indices = (complexity * (n_sizes - 1)).astype(np.int32)
        indices = np.clip(indices, 0, n_sizes - 1)

        kernel_size_map = self.KERNEL_SIZES[indices]

        unique, counts = np.unique(kernel_size_map, return_counts=True)
        self._kernel_stats = {
            "kernel_distribution": {int(k): int(c) for k, c in zip(unique, counts)},
            "mean_kernel_size": float(np.mean(kernel_size_map)),
            "complexity_mean": float(np.mean(complexity)),
            "complexity_std": float(np.std(complexity)),
        }

        return kernel_size_map, complexity

    def adaptive_morphological_tophat(self, gray, kernel_size_map):
        result = np.zeros_like(gray, dtype=np.float64)
        for ks in self.KERNEL_SIZES:
            mask = kernel_size_map == ks
            if not np.any(mask):
                continue
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
            result[mask] = tophat[mask].astype(np.float64)
        return result

    def adaptive_morphological_blackhat(self, gray, kernel_size_map):
        result = np.zeros_like(gray, dtype=np.float64)
        for ks in self.KERNEL_SIZES:
            mask = kernel_size_map == ks
            if not np.any(mask):
                continue
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            result[mask] = blackhat[mask].astype(np.float64)
        return result

    def adaptive_laplacian(self, gray, kernel_size_map):
        result = np.zeros_like(gray, dtype=np.float64)
        for ks in self.KERNEL_SIZES:
            mask = kernel_size_map == ks
            if not np.any(mask):
                continue
            sigma = ks / 4.0
            blur_ks = ks if ks % 2 == 1 else ks + 1
            blurred = cv2.GaussianBlur(gray.astype(np.float64), (blur_ks, blur_ks), sigma)
            lap = np.abs(cv2.Laplacian(blurred, cv2.CV_64F))
            result[mask] = lap[mask]
        return result

    def adaptive_dog(self, gray, kernel_size_map):
        result = np.zeros_like(gray, dtype=np.float64)
        for ks in self.KERNEL_SIZES:
            mask = kernel_size_map == ks
            if not np.any(mask):
                continue
            sigma1 = ks / 6.0
            sigma2 = ks / 2.0
            ks1 = max(3, int(np.ceil(sigma1 * 6)) | 1)
            ks2 = max(3, int(np.ceil(sigma2 * 6)) | 1)
            g1 = cv2.GaussianBlur(gray.astype(np.float64), (ks1, ks1), sigma1)
            g2 = cv2.GaussianBlur(gray.astype(np.float64), (ks2, ks2), sigma2)
            dog = np.abs(g1 - g2)
            result[mask] = dog[mask]
        return result

    def compute_adaptive_weights(self, gray, kernel_size_map, complexity_map, block_size=32):
        h, w = gray.shape
        w1 = np.ones((h, w), dtype=np.float64) * 0.30
        w2 = np.ones((h, w), dtype=np.float64) * 0.25
        w3 = np.ones((h, w), dtype=np.float64) * 0.25
        w4 = np.ones((h, w), dtype=np.float64) * 0.20

        gray_f = gray.astype(np.float64)
        local_mean = cv2.blur(gray_f, (block_size, block_size))
        local_var = self.compute_local_variance_map(gray, block_size)
        local_std = np.sqrt(np.clip(local_var, 0, None))

        bright_regions = local_mean > np.mean(local_mean) + 0.5 * np.std(local_mean)
        dark_regions = local_mean < np.mean(local_mean) - 0.5 * np.std(local_mean)
        high_texture = local_std > np.mean(local_std) + np.std(local_std)

        w1[bright_regions] = 0.40
        w2[dark_regions] = 0.40
        w3[high_texture] = 0.35
        w4[high_texture] = 0.30

        w1[dark_regions] = 0.15
        w2[bright_regions] = 0.15

        total = w1 + w2 + w3 + w4
        w1 /= total
        w2 /= total
        w3 /= total
        w4 /= total

        return w1, w2, w3, w4

    def compute_hybrid_response(self, gray, kernel_size_map, complexity_map):
        tophat = self.adaptive_morphological_tophat(gray, kernel_size_map)
        blackhat = self.adaptive_morphological_blackhat(gray, kernel_size_map)
        laplacian = self.adaptive_laplacian(gray, kernel_size_map)
        dog = self.adaptive_dog(gray, kernel_size_map)

        w1, w2, w3, w4 = self.compute_adaptive_weights(
            gray, kernel_size_map, complexity_map
        )

        response = w1 * tophat + w2 * blackhat + w3 * laplacian + w4 * dog

        return response, {
            "tophat": tophat,
            "blackhat": blackhat,
            "laplacian": laplacian,
            "dog": dog,
        }

    def hybrid_threshold(self, response):
        response_u8 = np.clip(response, 0, 255).astype(np.uint8)
        if response_u8.max() == response_u8.min():
            return np.zeros_like(response_u8)

        _, otsu = cv2.threshold(response_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(
            response_u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, -5
        )

        combined = cv2.bitwise_or(otsu, adaptive)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open, iterations=1)

        return combined

    def extract_defects(self, binary_mask, gray_original, min_area=30, max_area=50000):
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        img_h, img_w = gray_original.shape
        global_mean = float(np.mean(gray_original))
        global_std = float(np.std(gray_original))

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / max(h, 1)

            if aspect_ratio > 5.0 or aspect_ratio < 0.2:
                continue

            roi = gray_original[y:y + h, x:x + w]
            roi_mask = np.zeros((h, w), dtype=np.uint8)
            shifted = cnt.copy()
            shifted[:, 0, 0] -= x
            shifted[:, 0, 1] -= y
            cv2.drawContours(roi_mask, [shifted], -1, 255, -1)

            interior = roi[roi_mask == 255]
            if len(interior) == 0:
                continue

            interior_mean = float(np.mean(interior))
            interior_std = float(np.std(interior))

            perimeter = cv2.arcLength(cnt, True)
            circularity = 4.0 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

            if area < 500 and circularity > 0.3 and interior_mean < global_mean:
                class_id = 1
                defect_type = "small_pinhole"
            elif area >= 500 and interior_mean < global_mean - 0.3 * global_std:
                class_id = 2
                defect_type = "large_pinhole"
            elif interior_mean > global_mean + 0.5 * global_std:
                class_id = 0
                defect_type = "PbI2_excess"
            elif circularity > 0.25 and interior_mean < global_mean:
                if area < 500:
                    class_id = 1
                    defect_type = "small_pinhole"
                else:
                    class_id = 2
                    defect_type = "large_pinhole"
            else:
                continue

            contrast = abs(interior_mean - global_mean) / (global_std + 1e-6)
            size_score = np.clip(area / 2000.0, 0, 1)
            confidence = 0.6 * np.clip(contrast, 0, 1) + 0.25 * circularity + 0.15 * size_score
            confidence = float(np.clip(confidence, 0, 1))

            xc = (x + w / 2.0) / img_w
            yc = (y + h / 2.0) / img_h
            bw = w / img_w
            bh = h / img_h

            detections.append({
                "class_id": class_id,
                "defect_type": defect_type,
                "bbox_abs": (x, y, w, h),
                "bbox_yolo": (float(xc), float(yc), float(bw), float(bh)),
                "area": area,
                "circularity": circularity,
                "confidence": confidence,
                "interior_mean": interior_mean,
                "interior_std": interior_std,
            })

        return detections

    def get_kernel_stats(self):
        return self._kernel_stats.copy()
