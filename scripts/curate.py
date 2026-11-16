#!/usr/bin/env python3
"""Pick a diverse, deduplicated subset of the raw images for training.

Steps:
  1. Drop images listed in an optional exclude file (one filename or path per line).
  2. Drop images smaller than --min-side.
  3. Embed every image with CLIP and drop near-duplicates (cosine similarity >= --dup-threshold).
  4. Run k-means with --target clusters and keep the image nearest each cluster centre,
     so the subset covers the dataset's range of subjects instead of 300 bamboo stalks.
  5. Copy the kept images to <out>/images, write <out>/selection.jsonl and a contact sheet
     (<out>/contact_sheet_XX.jpg) to look over. Add any bad picks to the exclude file and rerun.

Usage:
    python scripts/curate.py --raw data/raw --out data/curated --target 300
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def load_rows(raw: Path) -> list[dict]:
    with open(raw / "metadata.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def embed(paths: list[Path], model_id: str, batch: int) -> np.ndarray:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = CLIPModel.from_pretrained(model_id, dtype=dtype).to(device).eval()
    proc = CLIPProcessor.from_pretrained(model_id)

    feats = []
    with torch.inference_mode():
        for i in range(0, len(paths), batch):
            imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
            inputs = proc(images=imgs, return_tensors="pt").to(device)
            # vision tower -> pooled output -> projection. Equivalent to get_image_features(), whose return type
            # changed from a tensor to a ModelOutput in newer transformers releases.
            pooled = model.vision_model(pixel_values=inputs["pixel_values"].to(dtype)).pooler_output
            f = model.visual_projection(pooled)
            f = torch.nn.functional.normalize(f.float(), dim=-1)
            feats.append(f.cpu().numpy())
            print(f"  embedded {min(i + batch, len(paths))}/{len(paths)}", end="\r")
    print()
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(feats)


def dedup(feats: np.ndarray, threshold: float) -> list[int]:
    keep: list[int] = []
    for i in range(len(feats)):
        if keep and float(np.max(feats[keep] @ feats[i])) >= threshold:
            continue
        keep.append(i)
    return keep


def contact_sheets(paths: list[Path], out: Path, cols: int = 10, thumb: int = 192, per_sheet: int = 100) -> None:
    for s in range(0, len(paths), per_sheet):
        chunk = paths[s:s + per_sheet]
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * thumb, rows * (thumb + 16)), "white")
        draw = ImageDraw.Draw(sheet)
        for j, p in enumerate(chunk):
            im = Image.open(p).convert("RGB")
            im.thumbnail((thumb, thumb))
            x, y = (j % cols) * thumb, (j // cols) * (thumb + 16)
            sheet.paste(im, (x, y))
            draw.text((x + 2, y + thumb + 1), p.name, fill="black")
        sheet.save(out / f"contact_sheet_{s // per_sheet:02d}.jpg", quality=88)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/curated")
    ap.add_argument("--target", type=int, default=300, help="number of images to keep (0 = keep all after dedup)")
    ap.add_argument("--dup-threshold", type=float, default=0.95)
    ap.add_argument("--min-side", type=int, default=512)
    ap.add_argument("--exclude", default=None, help="text file of filenames to drop")
    ap.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    rows = load_rows(raw)

    excluded: set[str] = set()
    if args.exclude and Path(args.exclude).exists():
        excluded = {Path(line.strip()).name for line in open(args.exclude) if line.strip()}
    rows = [r for r in rows if Path(r["file"]).name not in excluded]
    rows = [r for r in rows if min(r["width"], r["height"]) >= args.min_side]
    print(f"{len(rows)} images after exclude list ({len(excluded)}) and min-side {args.min_side}")

    paths = [raw / r["file"] for r in rows]
    feats = embed(paths, args.clip_model, args.batch)

    keep = dedup(feats, args.dup_threshold)
    print(f"{len(keep)} images after removing near-duplicates (threshold {args.dup_threshold})")
    rows = [rows[i] for i in keep]
    feats = feats[keep]

    if 0 < args.target < len(rows):
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=args.target, n_init=4, random_state=args.seed).fit(feats)
        chosen = []
        for c in range(args.target):
            members = np.where(km.labels_ == c)[0]
            d = np.linalg.norm(feats[members] - km.cluster_centers_[c], axis=1)
            chosen.append(int(members[np.argmin(d)]))
        rows = [rows[i] for i in sorted(chosen)]
        print(f"Selected {len(rows)} diverse images with k-means")

    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    kept_paths = []
    with open(out / "selection.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            src = raw / r["file"]
            dst = out / "images" / src.name
            shutil.copy2(src, dst)
            kept_paths.append(dst)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    contact_sheets(kept_paths, out)
    print(f"Wrote {len(rows)} images to {out / 'images'}; review {out}/contact_sheet_*.jpg")


if __name__ == "__main__":
    main()
