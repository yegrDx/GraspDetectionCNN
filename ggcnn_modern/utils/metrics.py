from __future__ import annotations
import numpy as np
import math
import cv2
from typing import List
from ..geometry.grasp import GraspRectangle


def polygon_iou(poly_a: np.ndarray, poly_b: np.ndarray, shape_hw) -> float:
    h, w = shape_hw
    ma = np.zeros((h, w), dtype=np.uint8)
    mb = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(ma, np.round(poly_a).astype(np.int32), 1)
    cv2.fillConvexPoly(mb, np.round(poly_b).astype(np.int32), 1)
    inter = np.logical_and(ma, mb).sum()
    union = np.logical_or(ma, mb).sum()
    return float(inter) / float(union) if union else 0.0


def angle_diff_rad(a: float, b: float) -> float:
    d = abs(a - b)
    return min(d, math.pi - d)


def grasp_success_iou(pred: GraspRectangle, gts: List[GraspRectangle], out_hw, iou_thresh: float = 0.25, angle_thresh_deg: float = 30.0) -> bool:
    _, _, a_pred = pred.width_height_angle
    for gt in gts:
        _, _, a_gt = gt.width_height_angle
        if angle_diff_rad(a_pred, a_gt) > math.radians(angle_thresh_deg):
            continue
        iou = polygon_iou(pred.corners, gt.corners, out_hw)
        if iou >= iou_thresh:
            return True
    return False
