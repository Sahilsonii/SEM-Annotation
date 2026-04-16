import cv2
import numpy as np
from scipy.ndimage import uniform_filter, generic_filter
from scipy.signal import fftconvolve


class ClassicalFilterBlend:

    @staticmethod
    def apply_clahe(gray, clip_limit=3.0, grid_size=8):
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
        return clahe.apply(gray)

    @staticmethod
    def morphological_tophat(gray, kernel_size=15):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    @staticmethod
    def morphological_blackhat(gray, kernel_size=15):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    @staticmethod
    def laplacian_of_gaussian(gray, sigma=2.0):
        ksize = int(np.ceil(sigma * 6)) | 1
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
        log = cv2.Laplacian(blurred, cv2.CV_64F)
        return np.abs(log).astype(np.float64)

    @staticmethod
    def difference_of_gaussian(gray, sigma1=1.0, sigma2=3.0):
        k1 = int(np.ceil(sigma1 * 6)) | 1
        k2 = int(np.ceil(sigma2 * 6)) | 1
        g1 = cv2.GaussianBlur(gray.astype(np.float64), (k1, k1), sigma1)
        g2 = cv2.GaussianBlur(gray.astype(np.float64), (k2, k2), sigma2)
        dog = g1 - g2
        return np.abs(dog)

    @staticmethod
    def gabor_filter_bank(gray, num_orientations=8, frequencies=None):
        if frequencies is None:
            frequencies = [0.05, 0.1, 0.2, 0.3]
        gray_f = gray.astype(np.float64)
        accumulator = np.zeros_like(gray_f)
        for freq in frequencies:
            sigma = 1.0 / (freq * np.pi) if freq > 0 else 1.0
            lambd = 1.0 / freq if freq > 0 else 10.0
            ksize = int(np.ceil(sigma * 6)) | 1
            if ksize < 3:
                ksize = 3
            for i in range(num_orientations):
                theta = i * np.pi / num_orientations
                kernel = cv2.getGaborKernel(
                    (ksize, ksize), sigma, theta, lambd, 0.5, 0, ktype=cv2.CV_64F
                )
                response = cv2.filter2D(gray_f, cv2.CV_64F, kernel)
                accumulator = np.maximum(accumulator, np.abs(response))
        return accumulator

    @staticmethod
    def hessian_blob_detector(gray, sigmas=None):
        if sigmas is None:
            sigmas = [1.0, 2.0, 4.0, 8.0]
        gray_f = gray.astype(np.float64)
        blob_response = np.zeros_like(gray_f)
        for sigma in sigmas:
            ksize = int(np.ceil(sigma * 6)) | 1
            if ksize < 3:
                ksize = 3
            smoothed = cv2.GaussianBlur(gray_f, (ksize, ksize), sigma)
            dxx = cv2.Sobel(smoothed, cv2.CV_64F, 2, 0, ksize=3)
            dyy = cv2.Sobel(smoothed, cv2.CV_64F, 0, 2, ksize=3)
            dxy = cv2.Sobel(smoothed, cv2.CV_64F, 1, 1, ksize=3)
            det_hessian = dxx * dyy - dxy * dxy
            scale_norm = sigma ** 2 * np.abs(det_hessian)
            blob_response = np.maximum(blob_response, scale_norm)
        return blob_response

    @staticmethod
    def local_binary_pattern_variance(gray, radius=3):
        gray_f = gray.astype(np.float64)
        h, w = gray_f.shape
        lbp_var = np.zeros_like(gray_f)
        offsets = []
        n_points = 8
        for i in range(n_points):
            angle = 2.0 * np.pi * i / n_points
            dy = int(round(radius * np.sin(angle)))
            dx = int(round(radius * np.cos(angle)))
            offsets.append((dy, dx))
        pad = radius + 1
        padded = np.pad(gray_f, pad, mode="reflect")
        neighbors = np.zeros((n_points, h, w), dtype=np.float64)
        for idx, (dy, dx) in enumerate(offsets):
            neighbors[idx] = padded[pad + dy:pad + dy + h, pad + dx:pad + dx + w]
        mean_neighbors = np.mean(neighbors, axis=0)
        lbp_var = np.var(neighbors, axis=0)
        contrast = np.abs(gray_f - mean_neighbors)
        return lbp_var + contrast

    @staticmethod
    def dct_highfreq_response(gray, block_size=32):
        h, w = gray.shape
        pad_h = (block_size - h % block_size) % block_size
        pad_w = (block_size - w % block_size) % block_size
        padded = np.pad(gray.astype(np.float32), ((0, pad_h), (0, pad_w)), mode="reflect")
        ph, pw = padded.shape
        response = np.zeros_like(padded, dtype=np.float32)
        cutoff = block_size // 3
        for r in range(0, ph, block_size):
            for c in range(0, pw, block_size):
                block = padded[r:r + block_size, c:c + block_size]
                dct_block = cv2.dct(block)
                mask = np.ones_like(dct_block)
                mask[:cutoff, :cutoff] = 0
                high_freq = cv2.idct(dct_block * mask)
                response[r:r + block_size, c:c + block_size] = np.abs(high_freq)
        return response[:h, :w].astype(np.float64)

    @staticmethod
    def unsharp_mask(gray, sigma=3.0, strength=2.0):
        ksize = int(np.ceil(sigma * 6)) | 1
        blurred = cv2.GaussianBlur(gray.astype(np.float64), (ksize, ksize), sigma)
        sharpened = gray.astype(np.float64) + strength * (gray.astype(np.float64) - blurred)
        return np.clip(sharpened, 0, 255)

    @staticmethod
    def median_deviation(gray, kernel_size=7):
        gray_f = gray.astype(np.float64)
        median = cv2.medianBlur(gray, kernel_size).astype(np.float64)
        return np.abs(gray_f - median)

    @staticmethod
    def local_entropy(gray, kernel_size=9):
        gray_f = gray.astype(np.float64)
        pad = kernel_size // 2
        padded = np.pad(gray_f, pad, mode="reflect")
        h, w = gray_f.shape
        entropy_map = np.zeros((h, w), dtype=np.float64)
        for r in range(h):
            for c in range(w):
                patch = padded[r:r + kernel_size, c:c + kernel_size].ravel()
                hist, _ = np.histogram(patch, bins=32, range=(0, 256))
                hist = hist / hist.sum()
                hist = hist[hist > 0]
                entropy_map[r, c] = -np.sum(hist * np.log2(hist))
        return entropy_map

    @staticmethod
    def local_entropy_fast(gray, kernel_size=9):
        gray_q = (gray.astype(np.float64) / 8.0).astype(np.int32)
        gray_q = np.clip(gray_q, 0, 31)
        h, w = gray.shape
        entropy_map = np.zeros((h, w), dtype=np.float64)
        pad = kernel_size // 2
        padded = np.pad(gray_q, pad, mode="reflect")
        area = kernel_size * kernel_size
        for r in range(h):
            if r == 0:
                hist = np.zeros(32, dtype=np.int32)
                patch = padded[0:kernel_size, 0:kernel_size]
                for val in patch.ravel():
                    hist[val] += 1
            else:
                old_row = padded[r - 1, 0:kernel_size]
                new_row = padded[r + kernel_size - 1, 0:kernel_size]
                for val in old_row.ravel():
                    hist[val] -= 1
                for val in new_row.ravel():
                    hist[val] += 1
            col_hist = hist.copy()
            probs = col_hist.astype(np.float64) / area
            valid = probs > 0
            entropy_map[r, 0] = -np.sum(probs[valid] * np.log2(probs[valid]))
            for c in range(1, w):
                old_col = padded[r:r + kernel_size, c - 1]
                new_col = padded[r:r + kernel_size, c + kernel_size - 1]
                for val in old_col:
                    col_hist[val] -= 1
                for val in new_col:
                    col_hist[val] += 1
                probs = col_hist.astype(np.float64) / area
                valid = probs > 0
                entropy_map[r, c] = -np.sum(probs[valid] * np.log2(probs[valid]))
        return entropy_map
