from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ggcnn_modern.models import build_model
from ggcnn_modern.datasets import CornellDataset, JacquardDataset
from ggcnn_modern.geometry.post_process import post_process_output
from ggcnn_modern.utils.metrics import grasp_success_iou
from ggcnn_modern.geometry.grasp import rects_from_cornell_file, rects_from_jacquard_file

def collate_keep_meta_list(batch):
    # batch: [(x, y_dict, meta_dict), ...]
    xs, ys, metas = zip(*batch)
    x = torch.stack(xs, dim=0)
    y = {k: torch.stack([yy[k] for yy in ys], dim=0) for k in ys[0].keys()}
    return x, y, list(metas)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True, choices=["cornell", "jacquard"])
    ap.add_argument("--index", required=True)
    ap.add_argument("--network", default=None, choices=[None, "ggcnn", "ggcnn2"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    net_name = args.network or cfg.get("network", "ggcnn")
    out_size = int(cfg.get("output_size", 300))
    width_norm = float(cfg.get("width_norm", 150.0))

    model = build_model(net_name).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if args.dataset == "cornell":
        ds = CornellDataset(args.index, output_size=out_size, width_norm=width_norm, seed=0)
        parse_gt = rects_from_cornell_file
    else:
        ds = JacquardDataset(args.index, output_size=out_size, width_norm=width_norm, seed=0)
        parse_gt = rects_from_jacquard_file

    n = min(args.n, len(ds))
    dl = DataLoader(Subset(ds, list(range(n))), batch_size=1, shuffle=False, collate_fn=collate_keep_meta_list)

    ok = 0
    for x, y, meta in tqdm(dl):
        x = x.to(device)
        pos, cos, sin, wid = model(x)
        pos = pos[0, 0].cpu().numpy()
        cos = cos[0, 0].cpu().numpy()
        sin = sin[0, 0].cpu().numpy()
        wid = wid[0, 0].cpu().numpy()

        pred = post_process_output(pos, cos, sin, wid, width_norm=width_norm)
        gpath = meta[0]["grasps"]
        gts = parse_gt(gpath)

        if grasp_success_iou(pred.rect, gts, out_hw=(out_size, out_size)):
            ok += 1

    print(f"Rect-metric success@{n}: {ok/n:.3f}")


if __name__ == "__main__":
    main()
