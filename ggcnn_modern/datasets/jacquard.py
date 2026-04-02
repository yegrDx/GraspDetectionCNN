from __future__ import annotations
import json
from .base import GraspDatasetBase
from ..geometry.grasp import rects_from_jacquard_file
from ..utils.image import load_depth


class JacquardDataset(GraspDatasetBase):
    def __init__(self, index_path: str, **kwargs):
        super().__init__(**kwargs)
        with open(index_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)
        if not isinstance(self.items, list) or len(self.items) == 0:
            raise ValueError("Jacquard index JSON must be a non-empty list.")

    def __len__(self) -> int:
        return len(self.items)

    def get_depth_and_rects(self, idx: int):
        it = self.items[idx]
        depth = load_depth(it["depth"])
        rects = rects_from_jacquard_file(it["grasps"])
        meta = dict(it)
        meta["orig_h"] = int(depth.shape[0])
        meta["orig_w"] = int(depth.shape[1])
        return depth, rects, meta
        return depth, rects, meta
