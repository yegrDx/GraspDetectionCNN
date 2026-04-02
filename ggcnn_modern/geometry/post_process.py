from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2
import math
from .grasp import GraspRectangle


@dataclass
class GraspPrediction:
    center_x: int
    center_y: int
    angle: float
    width_px: float
    score: float
    rect: GraspRectangle


def _argmax_2d(a: np.ndarray):
    idx = int(np.argmax(a))
    y, x = np.unravel_index(idx, a.shape)
    return int(y), int(x)


def post_process_output(
    q: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    width: np.ndarray,
    width_norm: float = 150.0,
    jaw_len_px: float = 30.0,
    smooth_sigma: float = 2.0,
) -> GraspPrediction:
    q_s = cv2.GaussianBlur(q.astype(np.float32), (0, 0), smooth_sigma) if smooth_sigma > 0 else q.astype(np.float32)
    y, x = _argmax_2d(q_s)
    score = float(q_s[y, x])

    ang = 0.5 * math.atan2(float(sin[y, x]), float(cos[y, x]))
    width_px = float(np.clip(width[y, x], 0.0, 1.0) * width_norm)

    rect = GraspRectangle.from_center_angle((x, y), ang, width=width_px, height=jaw_len_px)
    return GraspPrediction(center_x=x, center_y=y, angle=ang, width_px=width_px, score=score, rect=rect)
