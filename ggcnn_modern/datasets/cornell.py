from __future__ import annotations
import json
from .base import GraspDatasetBase
from ..geometry.grasp import rects_from_cornell_file
from ..utils.image import load_depth
from ..utils.pcd import depth_from_pcd


class CornellDataset(GraspDatasetBase):
    def __init__(self, index_path: str, **kwargs):
        super().__init__(**kwargs)
        with open(index_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)
        if not isinstance(self.items, list) or len(self.items) == 0:
            raise ValueError("Cornell index JSON must be a non-empty list.")

    def __len__(self) -> int:
        return len(self.items)

    def get_depth_and_rects(self, idx: int):
        it = self.items[idx]
        dpath = it["depth"]
        gpath = it["grasps"]
        if dpath.lower().endswith(".pcd"):
            depth = depth_from_pcd(dpath)
        else:
            depth = load_depth(dpath)
        rects = rects_from_cornell_file(gpath)
        meta = dict(it)
        return depth, rects, meta
