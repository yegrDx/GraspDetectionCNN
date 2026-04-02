from __future__ import annotations
import torch
from torch.utils.data import Dataset
import numpy as np
from ..geometry.labels import rects_to_maps
from ..utils.image import normalize_depth, resize_to_square
from ..geometry.grasp import GraspRectangle


class GraspDatasetBase(Dataset):
    def __init__(self, output_size: int = 300, width_norm: float = 150.0, gaussian: bool = True, seed: int = 0):
        self.output_size = int(output_size)
        self.width_norm = float(width_norm)
        self.gaussian = bool(gaussian)

    def __len__(self) -> int:
        raise NotImplementedError

    def get_depth_and_rects(self, idx: int):
        raise NotImplementedError

    def __getitem__(self, idx: int):
        depth, rects, meta = self.get_depth_and_rects(idx)

        # keep original shape for scaling rects
        H0, W0 = depth.shape[:2]
        if isinstance(meta, dict):
            H0 = int(meta.get("orig_h", H0))
            W0 = int(meta.get("orig_w", W0))

        # normalize + resize depth to square output
        depth = normalize_depth(depth)
        depth = resize_to_square(depth, self.output_size)

        # scale rectangles into output coordinates
        sx = self.output_size / float(W0)
        sy = self.output_size / float(H0)
        rects_s = []
        for r in rects:
            pts = r.corners.copy()
            pts[:, 0] *= sx
            pts[:, 1] *= sy
            rects_s.append(GraspRectangle(pts))

        pos, cos, sin, width = rects_to_maps(rects_s, self.output_size, width_norm=self.width_norm, gaussian=self.gaussian)

        x = torch.from_numpy(depth[None, ...].astype("float32"))
        y = {
            "pos": torch.from_numpy(pos[None, ...]),
            "cos": torch.from_numpy(cos[None, ...]),
            "sin": torch.from_numpy(sin[None, ...]),
            "width": torch.from_numpy(width[None, ...]),
        }
        return x, y, meta
