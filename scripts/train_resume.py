import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split
from tqdm import tqdm

# make project root importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

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


def build_dataset(dataset_name, index_paths, output_size, width_norm, gaussian, seed):
    if dataset_name == "cornell":
        return CornellDataset(index_paths[0], output_size=output_size, width_norm=width_norm, gaussian=gaussian, seed=seed)
    if dataset_name == "jacquard":
        return JacquardDataset(index_paths[0], output_size=output_size, width_norm=width_norm, gaussian=gaussian, seed=seed)
    if dataset_name == "mixed":
        if len(index_paths) < 2:
            raise SystemExit("mixed requires: --index cornell_index.json jacquard_index.json")
        ds1 = CornellDataset(index_paths[0], output_size=output_size, width_norm=width_norm, gaussian=gaussian, seed=seed)
        ds2 = JacquardDataset(index_paths[1], output_size=output_size, width_norm=width_norm, gaussian=gaussian, seed=seed + 1)
        return ConcatDataset([ds1, ds2])
    raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", required=True, help="Path to checkpoint (last.pt or best.pt)")
    ap.add_argument("--outdir", default=None, help="If set, override outdir. Otherwise use checkpoint cfg outdir if possible.")
    ap.add_argument("--epochs", type=int, default=None, help="Total epochs to train to. If not set, uses checkpoint cfg epochs.")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--reset-optim", action="store_true", help="Ignore optimizer state from checkpoint (start with fresh optimizer).")
    ap.add_argument("--seed", type=int, default=None, help="Override seed (optional).")
    args = ap.parse_args()

    ckpt = torch.load(args.resume, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    if not cfg:
        raise SystemExit("Checkpoint does not contain cfg. Use your original train.py with same args, or re-save a checkpoint with cfg.")

    # take training params from checkpoint cfg, allow overrides
    network = cfg.get("network", "ggcnn")
    dataset_name = cfg.get("dataset", "cornell")
    index_paths = cfg.get("index", [])
    outdir = args.outdir or cfg.get("outdir", os.path.join("output", "resume_run"))
    total_epochs = args.epochs if args.epochs is not None else int(cfg.get("epochs", 50))
    batch_size = args.batch_size if args.batch_size is not None else int(cfg.get("batch_size", 8))
    lr = args.lr if args.lr is not None else float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    output_size = int(cfg.get("output_size", 300))
    width_norm = float(cfg.get("width_norm", 150.0))
    gaussian = not bool(cfg.get("no_gaussian", False))
    val_split = float(cfg.get("val_split", 0.1))
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 0))
    num_workers = args.num_workers if args.num_workers is not None else int(cfg.get("num_workers", 2))

    os.makedirs(outdir, exist_ok=True)
    seed_everything(seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # rebuild model/dataset
    model = build_model(network).to(device)
    model.load_state_dict(ckpt["model_state"])

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if (not args.reset_optim) and ("optimizer_state" in ckpt):
        try:
            opt.load_state_dict(ckpt["optimizer_state"])
        except Exception:
            print("WARN: could not load optimizer_state, continuing with fresh optimizer.")
    elif args.reset_optim:
        print("INFO: reset_optim enabled -> using fresh optimizer state.")

    ds = build_dataset(dataset_name, index_paths, output_size, width_norm, gaussian, seed)

    # IMPORTANT: to keep split consistent across resume, we use the same seed as before.
    n_val = max(1, int(len(ds) * val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_val = float(ckpt.get("best_val", float("inf")))

    # save updated config for traceability
    new_cfg = dict(cfg)
    new_cfg["outdir"] = outdir
    new_cfg["epochs"] = total_epochs
    new_cfg["batch_size"] = batch_size
    new_cfg["lr"] = lr
    new_cfg["seed"] = seed
    new_cfg["num_workers"] = num_workers
    with open(os.path.join(outdir, "config_resume.json"), "w", encoding="utf-8") as f:
        json.dump(new_cfg, f, ensure_ascii=False, indent=2)

    print(f"Resuming from: {args.resume}")
    print(f"Start epoch: {start_epoch} / Target: {total_epochs}")
    print(f"Best val so far: {best_val:.6f}")
    print(f"Outdir: {outdir}")

    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        losses = []
        for x, y, _ in tqdm(train_dl, desc=f"epoch {epoch}/{total_epochs}", leave=False):
            x = x.to(device)
            y = {k: v.to(device) for k, v in y.items()}
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            l = loss_fn(pred, y)
            l.backward()
            opt.step()
            losses.append(l.item())

        val = validate(model, val_dl, device)
        print(f"Epoch {epoch:03d} train_loss={np.mean(losses):.6f} val_loss={val:.6f}")

        # save last
        save_checkpoint(os.path.join(outdir, "last.pt"), model, opt, epoch, best_val, new_cfg)

        # save best
        if val < best_val:
            best_val = val
            save_checkpoint(os.path.join(outdir, "best.pt"), model, opt, epoch, best_val, new_cfg)
            print(f"  -> new best: {best_val:.6f}")

    print("Done. Best val:", best_val)


if __name__ == "__main__":
    main()
