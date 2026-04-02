from __future__ import annotations
import numpy as np
import cv2


def load_depth(path: str) -> np.ndarray:
    if path.lower().endswith(".npy"):
        return np.load(path).astype(np.float32)
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.ndim == 3:
        im = im[:, :, 0]
    return im.astype(np.float32)


def inpaint_depth(depth: np.ndarray) -> np.ndarray:
    d = depth.copy().astype(np.float32)
    mask = (~np.isfinite(d)) | (d <= 0)
    d[~np.isfinite(d)] = 0.0
    if mask.any():
        v = d.copy()
        v[mask] = 0
        p2 = np.percentile(v[v > 0], 95) if np.any(v > 0) else 1.0
        scale = 255.0 / max(p2, 1e-6)
        v8 = np.clip(v * scale, 0, 255).astype(np.uint8)
        mask8 = (mask.astype(np.uint8) * 255)
        v8 = cv2.inpaint(v8, mask8, 3, cv2.INPAINT_NS)
        d = v8.astype(np.float32) / scale
    return d


def normalize_depth(depth: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    d = inpaint_depth(depth.astype(np.float32))
    valid = d[d > 0]
    if valid.size == 0:
        return np.zeros_like(d, dtype=np.float32)
    lo = np.percentile(valid, 5)
    hi = np.percentile(valid, 95)
    if hi - lo < eps:
        return np.zeros_like(d, dtype=np.float32)
    d = np.clip(d, lo, hi)
    d = (d - lo) / (hi - lo)
    d = d * 2.0 - 1.0
    return d


def resize_to_square(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
