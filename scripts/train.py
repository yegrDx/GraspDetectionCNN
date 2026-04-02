from __future__ import annotations
import argparse
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler, random_split
from tqdm import tqdm

from ggcnn_modern.models import build_model
from ggcnn_modern.datasets import CornellDataset, JacquardDataset
from ggcnn_modern.utils.train_utils import seed_everything, save_checkpoint


def loss_fn(pred, target):
    p_pos, p_cos, p_sin, p_wid = pred
    return (
        F.mse_loss(p_pos, target["pos"])
        + F.mse_loss(p_cos, target["cos"])
        + F.mse_loss(p_sin, target["sin"])
        + F.mse_loss(p_wid, target["width"])
    )
def collate_keep_meta_list(batch):
    # batch: [(x, y_dict, meta_dict), ...]
    xs, ys, metas = zip(*batch)
    x = torch.stack(xs, dim=0)
    y = {k: torch.stack([yy[k] for yy in ys], dim=0) for k in ys[0].keys()}
    return x, y, list(metas)

@torch.no_grad()
def validate(model, dl, device):
    model.eval()
    losses = []
    for x, y, _ in dl:
        x = x.to(device)
        y = {k: v.to(device) for k, v in y.items()}
        pred = model(x)
        losses.append(loss_fn(pred, y).item())
    return float(np.mean(losses)) if losses else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="ggcnn", choices=["ggcnn", "ggcnn2"])
    ap.add_argument("--dataset", default="cornell", choices=["cornell", "jacquard", "mixed"])
    ap.add_argument("--index", nargs="+", required=True, help="Index JSON path(s). For mixed: two paths.")
    ap.add_argument("--outdir", default="output/run")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--output-size", type=int, default=300)
    ap.add_argument("--width-norm", type=float, default=150.0)
    ap.add_argument("--no-gaussian", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--mix-weights", nargs="+", type=float, default=[0.5, 0.5])
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    seed_everything(args.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = build_model(args.network).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    cfg = vars(args)
    with open(os.path.join(args.outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    if args.dataset == "cornell":
        ds = CornellDataset(args.index[0], output_size=args.output_size, width_norm=args.width_norm, gaussian=not args.no_gaussian, seed=args.seed)
    elif args.dataset == "jacquard":
        ds = JacquardDataset(args.index[0], output_size=args.output_size, width_norm=args.width_norm, gaussian=not args.no_gaussian, seed=args.seed)
    else:
        if len(args.index) < 2:
            raise SystemExit("mixed requires: --index cornell_index.json jacquard_index.json")
        ds1 = CornellDataset(args.index[0], output_size=args.output_size, width_norm=args.width_norm, gaussian=not args.no_gaussian, seed=args.seed)
        ds2 = JacquardDataset(args.index[1], output_size=args.output_size, width_norm=args.width_norm, gaussian=not args.no_gaussian, seed=args.seed + 1)
        ds = ConcatDataset([ds1, ds2])

    n_val = max(1, int(len(ds) * args.val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_keep_meta_list,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_keep_meta_list,
    )

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y, _ in tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            x = x.to(device)
            y = {k: v.to(device) for k, v in y.items()}
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            l = loss_fn(pred, y)
            l.backward()
            opt.step()
            losses.append(l.item())

        val = validate(model, val_dl, device)
        print(f"Epoch {epoch:03d} train_loss={np.mean(losses):.5f} val_loss={val:.5f}")

        save_checkpoint(os.path.join(args.outdir, "last.pt"), model, opt, epoch, best, cfg)
        if val < best:
            best = val
            save_checkpoint(os.path.join(args.outdir, "best.pt"), model, opt, epoch, best, cfg)
            print(f"  -> new best: {best:.5f}")

    print("Done. Best val:", best)


if __name__ == "__main__":
    main()
