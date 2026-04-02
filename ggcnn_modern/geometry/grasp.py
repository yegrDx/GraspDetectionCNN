from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple
import math
import numpy as np
import cv2


def _to_np(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    if a.shape != (4, 2):
        raise ValueError(f"Expected (4,2) corners, got {a.shape}")
    return a


@dataclass
class GraspRectangle:
    """
    Прямоугольник захвата в координатах изображения (x вправо, y вниз).
    corners: 4 точки (x,y) по периметру.
    """
    corners: np.ndarray  # (4,2)

    @staticmethod
    def from_corners(corners: Iterable[Iterable[float]]) -> "GraspRectangle":
        return GraspRectangle(_to_np(corners))

    @staticmethod
    def from_center_angle(center_xy: Tuple[float, float], angle_rad: float, width: float, height: float) -> "GraspRectangle":
        cx, cy = center_xy
        ca = math.cos(angle_rad)
        sa = math.sin(angle_rad)
        u = np.array([ca, sa], dtype=np.float32)
        v = np.array([-sa, ca], dtype=np.float32)
        w2 = width / 2.0
        h2 = height / 2.0
        c = np.array([cx, cy], dtype=np.float32)

        pts = np.stack([
            c - u * w2 - v * h2,
            c + u * w2 - v * h2,
            c + u * w2 + v * h2,
            c - u * w2 + v * h2,
        ], axis=0)
        return GraspRectangle(pts)

    @property
    def center(self) -> Tuple[float, float]:
        c = self.corners.mean(axis=0)
        return float(c[0]), float(c[1])

    def as_int(self) -> np.ndarray:
        return np.round(self.corners).astype(np.int32)

    def min_area_rect(self):
        rect = cv2.minAreaRect(self.corners.astype(np.float32))
        return rect

    @property
    def width_height_angle(self) -> Tuple[float, float, float]:
        (cx, cy), (w, h), a = self.min_area_rect()
        # Gripper opening = short side. Angle is along long axis.
        if w < h:
            angle = math.radians(a) + math.pi / 2
            opening = w
            jaw_len = h
        else:
            angle = math.radians(a)
            opening = h
            jaw_len = w
        while angle < -math.pi/2:
            angle += math.pi
        while angle > math.pi/2:
            angle -= math.pi
        return float(opening), float(jaw_len), float(angle)

    def rotate(self, angle_rad: float, center: Tuple[float, float]) -> "GraspRectangle":
        cx, cy = center
        ca = math.cos(angle_rad)
        sa = math.sin(angle_rad)
        pts = self.corners.copy()
        pts[:, 0] -= cx
        pts[:, 1] -= cy
        x = pts[:, 0] * ca - pts[:, 1] * sa
        y = pts[:, 0] * sa + pts[:, 1] * ca
        pts[:, 0] = x + cx
        pts[:, 1] = y + cy
        return GraspRectangle(pts)

    def scale(self, s: float, center: Tuple[float, float]) -> "GraspRectangle":
        cx, cy = center
        pts = self.corners.copy()
        pts[:, 0] = (pts[:, 0] - cx) * s + cx
        pts[:, 1] = (pts[:, 1] - cy) * s + cy
        return GraspRectangle(pts)

    def offset(self, dx: float, dy: float) -> "GraspRectangle":
        pts = self.corners.copy()
        pts[:, 0] += dx
        pts[:, 1] += dy
        return GraspRectangle(pts)


def rects_from_cornell_file(path: str) -> List[GraspRectangle]:
    rects: List[GraspRectangle] = []
    pts: List[Tuple[float, float]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0]); y = float(parts[1])
            except ValueError:
                continue
            if not (math.isfinite(x) and math.isfinite(y)):
                pts = []
                continue
            pts.append((x, y))
            if len(pts) == 4:
                rects.append(GraspRectangle.from_corners(pts))
                pts = []
    return rects


def rects_from_jacquard_file(path: str) -> List[GraspRectangle]:
    rects: List[GraspRectangle] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 5:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
                theta_deg = float(parts[2])
                opening = float(parts[3])
                jaw = float(parts[4])
            except ValueError:
                continue
            # Jacquard doc note: angle is horizontally mirrored in image coords -> negate.
            theta = -math.radians(theta_deg)
            rects.append(GraspRectangle.from_center_angle((x, y), theta, width=opening, height=jaw))
    return rects


def draw_rectangles(img: np.ndarray, rects: List[GraspRectangle], color=(0, 255, 0), thickness=2) -> np.ndarray:
    out = img.copy()
    for r in rects:
        cv2.polylines(out, [r.as_int()], isClosed=True, color=color, thickness=thickness)
    return out
