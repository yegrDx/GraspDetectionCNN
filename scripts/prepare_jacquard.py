
import argparse
import json
import os
import glob
from tqdm import tqdm


DEPTH_SUFFIXES_PERFECT_FIRST = [
    "_perfect_depth.tiff",
    "_perfect_depth.tif",
    "_stereo_depth.tiff",
    "_stereo_depth.tif",
    "_depth.png",
    "_depth.tiff",
    "_depth.tif",
]

DEPTH_SUFFIXES_STEREO_FIRST = [
    "_stereo_depth.tiff",
    "_stereo_depth.tif",
    "_perfect_depth.tiff",
    "_perfect_depth.tif",
    "_depth.png",
    "_depth.tiff",
    "_depth.tif",
]


def pick_existing(base: str, suffixes: list[str]) -> str | None:
    for s in suffixes:
        p = base + s
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", required=True, help=r"E:\datasets\Jacquard")
    ap.add_argument("--output-index", required=True, help=r"indexes\jacquard_index.json")
    ap.add_argument("--prefer-depth", choices=["perfect", "stereo"], default="perfect")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit, else stop after N samples (debug)")
    ap.add_argument("--no-recursive", action="store_true", help="If your structure is flat, disable recursion (usually keep off)")
    args = ap.parse_args()

    root = os.path.abspath(args.dataset_path)

    pattern = os.path.join(root, "*_grasps.txt") if args.no_recursive else os.path.join(root, "**", "*_grasps.txt")
    grasp_files = glob.glob(pattern, recursive=not args.no_recursive)
    grasp_files.sort()

    if not grasp_files:
        raise SystemExit("Не найдено *_grasps.txt. Проверь путь dataset-path.")

    suffixes = DEPTH_SUFFIXES_PERFECT_FIRST if args.prefer_depth == "perfect" else DEPTH_SUFFIXES_STEREO_FIRST

    items = []
    missing_depth = 0

    it = tqdm(grasp_files, desc="Indexing Jacquard", unit="file")
    for g in it:
        base = g[:-len("_grasps.txt")]  # ".../0_<id>" или ".../1_<id>" и т.п.
        depth = pick_existing(base, suffixes)
        if depth is None:
            missing_depth += 1
            continue

        # optional, but handy
        rgb = None
        for s in ["_RGB.png", "_rgb.png", "_RGB.jpg", "_RGB.jpeg"]:
            p = base + s
            if os.path.exists(p):
                rgb = p
                break

        mask = None
        for s in ["_mask.png", "_mask.tiff", "_mask.tif"]:
            p = base + s
            if os.path.exists(p):
                mask = p
                break

        item = {
            "depth": depth,
            "grasps": g,
            "orig_w": None,
            "orig_h": None,
        }
        if rgb:
            item["rgb"] = rgb
        if mask:
            item["mask"] = mask

        items.append(item)

        if args.limit and len(items) >= args.limit:
            break

        it.set_postfix(found=len(items), missing_depth=missing_depth)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_index)), exist_ok=True)
    with open(args.output_index, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(items)} samples to {args.output_index}")
    print(f"Missing depth for {missing_depth} grasp files (skipped)")
    if items:
        print("Example item:\n", json.dumps(items[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
