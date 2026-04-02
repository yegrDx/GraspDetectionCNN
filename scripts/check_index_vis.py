# scripts/check_index_vis.py
from __future__ import annotations

import os
import sys
import json
import argparse
import random
from collections import Counter

# Make project root importable (Windows-friendly when running as script)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import cv2

from ggcnn_modern.utils.image import load_depth
from ggcnn_modern.utils.pcd import depth_from_pcd
from ggcnn_modern.geometry.grasp import (
    rects_from_cornell_file,
    rects_from_jacquard_file,
    draw_rectangles,
)


def depth_to_vis(depth: np.ndarray) -> np.ndarray:
    """Convert depth array to a nice 8-bit grayscale visualization."""
    d = depth.astype(np.float32).copy()
    d[~np.isfinite(d)] = 0.0

    # treat <=0 as invalid
    valid = d[d > 0]
    if valid.size == 0:
        vis = np.zeros_like(d, dtype=np.uint8)
        return vis

    lo = np.percentile(valid, 2)
    hi = np.percentile(valid, 98)
    if hi - lo < 1e-6:
        vis = np.zeros_like(d, dtype=np.uint8)
        return vis

    d = np.clip(d, lo, hi)
    d = (d - lo) / (hi - lo)  # 0..1
    vis = (d * 255).astype(np.uint8)
    return vis


def detect_dataset_type(item: dict) -> str:
    g = str(item.get("grasps", ""))
    if g.endswith("cpos.txt"):
        return "cornell"
    if g.endswith("_grasps.txt"):
        return "jacquard"
    return "unknown"


def load_depth_any(path: str) -> np.ndarray:
    if path.lower().endswith(".pcd"):
        return depth_from_pcd(path)
    return load_depth(path)


def parse_rects(dataset: str, grasp_path: str):
    if dataset == "cornell":
        return rects_from_cornell_file(grasp_path)
    if dataset == "jacquard":
        return rects_from_jacquard_file(grasp_path)
    raise ValueError(f"Unknown dataset type: {dataset}")


def rects_out_of_bounds(rects, w: int, h: int) -> int:
    bad = 0
    for r in rects:
        pts = r.corners
        if (
            np.any(pts[:, 0] < 0)
            or np.any(pts[:, 0] >= w)
            or np.any(pts[:, 1] < 0)
            or np.any(pts[:, 1] >= h)
        ):
            bad += 1
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="Path to index JSON (cornell_index.json or jacquard_index.json)")
    ap.add_argument("--dataset", default="auto", choices=["auto", "cornell", "jacquard"])
    ap.add_argument("--outdir", default="check_vis", help="Where to save visualizations")
    ap.add_argument("--num", type=int, default=50, help="How many samples to visualize (random). Use 0 for all.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-grasps", type=int, default=200, help="Limit number of rectangles drawn per sample (Jacquard can be huge)")
    ap.add_argument("--thickness", type=int, default=2)
    ap.add_argument("--save-bad-only", action="store_true", help="Save only samples with problems (missing/empty/oob/etc.)")
    ap.add_argument("--show", action="store_true", help="Show OpenCV window (press any key to step, ESC to quit)")
    args = ap.parse_args()

    with open(args.index, "r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list) or len(items) == 0:
        raise SystemExit("Index must be a non-empty JSON list.")

    os.makedirs(args.outdir, exist_ok=True)
    rnd = random.Random(args.seed)

    if args.num == 0 or args.num >= len(items):
        idxs = list(range(len(items)))
    else:
        idxs = rnd.sample(range(len(items)), k=args.num)

    stats = Counter()
    saved = 0

    for i, idx in enumerate(idxs, 1):
        it = items[idx]
        dpath = it.get("depth", "")
        gpath = it.get("grasps", "")

        dataset = args.dataset
        if dataset == "auto":
            dataset = detect_dataset_type(it)

        problems = []

        if dataset == "unknown":
            problems.append("unknown_dataset_type")

        if not dpath or not os.path.exists(dpath):
            problems.append("missing_depth")
        if not gpath or not os.path.exists(gpath):
            problems.append("missing_grasps")

        depth = None
        rects = []

        # load depth
        if "missing_depth" not in problems:
            try:
                depth = load_depth_any(dpath)
            except Exception as e:
                problems.append(f"depth_read_error:{type(e).__name__}")

        # parse grasps
        if "missing_grasps" not in problems and dataset != "unknown":
            try:
                rects = parse_rects(dataset, gpath)
                if len(rects) == 0:
                    problems.append("empty_grasps")
            except Exception as e:
                problems.append(f"grasps_parse_error:{type(e).__name__}")

        if depth is not None and rects:
            h, w = depth.shape[:2]
            oob = rects_out_of_bounds(rects, w, h)
            if oob > 0:
                problems.append(f"oob_rects:{oob}")

        # update stats
        stats["total_checked"] += 1
        if problems:
            stats["bad_samples"] += 1
            for p in problems:
                stats[p] += 1
        else:
            stats["good_samples"] += 1

        # decide saving
        if args.save_bad_only and not problems:
            continue
        if depth is None:
            continue  # nothing to visualize

        # visualize
        vis = depth_to_vis(depth)
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        if rects:
            if len(rects) > args.max_grasps:
                rects_draw = rects[: args.max_grasps]
                stats["grasp_truncated"] += 1
            else:
                rects_draw = rects
            vis = draw_rectangles(vis, rects_draw, color=(0, 255, 0), thickness=args.thickness)

        # overlay text
        txt1 = f"{dataset} | idx={idx} | rects={len(rects)}"
        txt2 = "OK" if not problems else ("; ".join(problems)[:120])
        cv2.putText(vis, txt1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, txt2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if problems else (0, 255, 0), 2, cv2.LINE_AA)

        out_path = os.path.join(args.outdir, f"{dataset}_{idx:06d}.png")
        cv2.imwrite(out_path, vis)
        saved += 1

        if args.show:
            cv2.imshow("check_index_vis", vis)
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC
                break

    if args.show:
        cv2.destroyAllWindows()

    print("\n=== DONE ===")
    print("Index:", args.index)
    print("Saved images:", saved)
    print("Stats:")
    for k, v in stats.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
