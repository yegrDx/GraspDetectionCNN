from __future__ import annotations
import argparse
import json
import os
import cv2
import numpy as np
import torch

from ggcnn_modern.models import build_model
from ggcnn_modern.utils.image import load_depth, normalize_depth, resize_to_square
from ggcnn_modern.geometry.post_process import post_process_output
from ggcnn_modern.geometry.grasp import draw_rectangles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--depth", required=True)
    ap.add_argument("--network", default=None, choices=[None, "ggcnn", "ggcnn2"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--width-norm", type=float, default=150.0)
    ap.add_argument("--output-size", type=int, default=300)
    ap.add_argument("--vis", default=None)
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    net_name = args.network or cfg.get("network", "ggcnn")
    out_size = int(cfg.get("output_size", args.output_size))
    width_norm = float(cfg.get("width_norm", args.width_norm))

    model = build_model(net_name).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    depth = load_depth(args.depth)
    depth_n = normalize_depth(depth)
    depth_n = resize_to_square(depth_n, out_size)

    x = torch.from_numpy(depth_n[None, None, ...].astype(np.float32)).to(device)
    with torch.no_grad():
        pos, cos, sin, wid = model(x)

    pos = pos[0, 0].cpu().numpy()
    cos = cos[0, 0].cpu().numpy()
    sin = sin[0, 0].cpu().numpy()
    wid = wid[0, 0].cpu().numpy()

    pred = post_process_output(pos, cos, sin, wid, width_norm=width_norm)

    print(json.dumps({
        "center_x": pred.center_x,
        "center_y": pred.center_y,
        "angle_rad": pred.angle,
        "width_px": pred.width_px,
        "score": pred.score,
    }, ensure_ascii=False, indent=2))

    if args.vis:
        vis = (depth_n - depth_n.min()) / (depth_n.max() - depth_n.min() + 1e-6)
        vis = (vis * 255).astype(np.uint8)
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        vis = draw_rectangles(vis, [pred.rect], color=(0, 255, 0), thickness=2)
        os.makedirs(os.path.dirname(os.path.abspath(args.vis)), exist_ok=True)
        cv2.imwrite(args.vis, vis)
        print("Saved vis to:", args.vis)


if __name__ == "__main__":
    main()
