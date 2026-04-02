from __future__ import annotations
from typing import List, Tuple
import numpy as np
import cv2
import math
from .grasp import GraspRectangle


def _mask_from_rect(rect: GraspRectangle, out_hw: Tuple[int, int]) -> np.ndarray:
    h, w = out_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, rect.as_int(), 1)
    return mask.astype(bool)


def rects_to_maps(
    rects: List[GraspRectangle],
    out_size: int,
    width_norm: float = 150.0,
    gaussian: bool = True,
    sigma: float = 2.0,
):
    h = w = out_size
    pos = np.zeros((h, w), dtype=np.float32)
    cos = np.zeros((h, w), dtype=np.float32)
    sin = np.zeros((h, w), dtype=np.float32)
    width = np.zeros((h, w), dtype=np.float32)

    for r in rects:
        opening, jaw, ang = r.width_height_angle
        m = _mask_from_rect(r, (h, w))
        if not np.any(m):
            continue
        vpos = 1.0
        vcos = math.cos(2 * ang)
        vsin = math.sin(2 * ang)
        vwid = np.clip(opening / float(width_norm), 0.0, 1.0)

        update = m & (vpos > pos)
        pos[update] = vpos
        cos[update] = vcos
        sin[update] = vsin
        width[update] = vwid

    if gaussian and pos.max() > 0:
        pos = cv2.GaussianBlur(pos, ksize=(0, 0), sigmaX=sigma)
        mx = float(pos.max())
        if mx > 1e-6:
            pos = pos / mx

    return pos, cos, sin, width
