import argparse, json, random, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-index", required=True)
    ap.add_argument("--out-index", required=True)
    ap.add_argument("--n", type=int, required=True, help="how many samples to keep")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true", help="randomly sample instead of taking first N")
    args = ap.parse_args()

    with open(args.in_index, "r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise SystemExit("Index must be a JSON list")

    n = min(args.n, len(items))
    if args.shuffle:
        rnd = random.Random(args.seed)
        items = rnd.sample(items, n)
    else:
        items = items[:n]

    os.makedirs(os.path.dirname(os.path.abspath(args.out_index)), exist_ok=True)
    with open(args.out_index, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(items)} items to {args.out_index}")

if __name__ == "__main__":
    main()
