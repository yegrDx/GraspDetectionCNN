from __future__ import annotations
import argparse
import json
import os
import glob
import cv2
import numpy as np
from ggcnn_modern.utils.pcd import depth_from_pcd


def find_cornell_samples(root: str):
    grasp_files = glob.glob(os.path.join(root, "**", "*cpos.txt"), recursive=True)
    grasp_files.sort()
    samples = []
    for g in grasp_files:
        base = g[:-len("cpos.txt")]
        depth_candidates = [
            base + "d.png",
            base + "d.tiff",
            base + "d.tif",
            base + "d.npy",
            base + ".pcd",
        ]
        dpath = None
        for c in depth_candidates:
            if os.path.exists(c):
                dpath = c
                break
        if dpath is None:
            continue
        samples.append((dpath, g))
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--output-index", required=True)
    ap.add_argument("--convert-pcd", action="store_true", help="Convert .pcd -> *_depth.npy (float32) and index npy")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(args.dataset_path)
    samples = find_cornell_samples(root)
    if not samples:
        raise SystemExit("No Cornell samples found (did not find *cpos.txt).")

    items = []
    for dpath, gpath in samples:
        dpath2 = dpath
        if args.convert_pcd and dpath.lower().endswith(".pcd"):
            npy_path = dpath[:-4] + "_depth.npy"
            if (not os.path.exists(npy_path)) or args.overwrite:
                depth = depth_from_pcd(dpath)
                np.save(npy_path, depth.astype(np.float32))
            dpath2 = npy_path

        if dpath2.lower().endswith(".npy"):
            depth = np.load(dpath2)
        elif dpath2.lower().endswith(".pcd"):
            depth = depth_from_pcd(dpath2)
        else:
            im = cv2.imread(dpath2, cv2.IMREAD_UNCHANGED)
            if im is None:
                continue
            if im.ndim == 3:
                im = im[:, :, 0]
            depth = im
        h, w = depth.shape[:2]

        items.append({"depth": dpath2, "grasps": gpath, "orig_w": int(w), "orig_h": int(h)})

    os.makedirs(os.path.dirname(os.path.abspath(args.output_index)), exist_ok=True)
    with open(args.output_index, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(items)} samples to {args.output_index}")


if __name__ == "__main__":
    main()
