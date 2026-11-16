#!/usr/bin/env python3
"""Download the Sumi-e dataset from Hugging Face and save it as image files + metadata.

Output layout:
    <out_dir>/images/00000.png ...
    <out_dir>/metadata.jsonl   one row per image: file, width, height, seed, positive_prompt, model, ...

Usage:
    python scripts/download_dataset.py --out data/raw
"""

import argparse
import json
from collections import Counter
from pathlib import Path

DATASET_ID = "takara-ai/Sumi-e_no_kurozuappu_ink_painting_closeups"
META_FIELDS = [
    "seed", "positive_prompt", "negative_prompt", "model", "steps",
    "cfg", "sampler_name", "scheduler", "denoise",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DATASET_ID)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()

    from datasets import load_dataset  # imported here so --help works without it

    out = Path(args.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, split=args.split)
    print(f"Loaded {len(ds)} rows. Columns: {ds.column_names}")

    sizes: Counter = Counter()
    models: Counter = Counter()
    with open(out / "metadata.jsonl", "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            img = row["image"].convert("RGB")
            name = f"{i:05d}.png"
            img.save(img_dir / name)
            rec = {"file": f"images/{name}", "width": img.width, "height": img.height}
            for k in META_FIELDS:
                if k in row:
                    v = row[k]
                    rec[k] = v if isinstance(v, (str, int, float, bool)) or v is None else str(v)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            sizes[(img.width, img.height)] += 1
            models[str(rec.get("model"))] += 1
            if (i + 1) % 200 == 0:
                print(f"  saved {i + 1}/{len(ds)}")

    print("\nImage sizes (w x h: count):")
    for (w, h), n in sizes.most_common():
        print(f"  {w} x {h}: {n}")
    print("\nGenerator models used:")
    for m, n in models.most_common():
        print(f"  {m}: {n}")
    print(f"\nWrote {sum(sizes.values())} images to {img_dir} and metadata to {out / 'metadata.jsonl'}")


if __name__ == "__main__":
    main()
