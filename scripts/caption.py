#!/usr/bin/env python3
"""Write natural-language captions and the trainer's dataset.json.

Two modes:
  --mode vlm     (recommended) Caption each image with a vision-language model (Qwen2.5-VL by default).
                 The original Stable Diffusion prompt is passed in as a hint for the subject only.
  --mode prompt  No GPU needed: clean up the dataset's positive_prompt (strip weights, quality tags,
                 artist/style boilerplate) and turn it into a sentence. Quicker, but captions are weaker.

Do NOT put the trigger word in captions: the trainer prepends `lora_trigger` itself during preprocessing.

Output: <data_dir>/dataset.json  ->  [{"video": "images/00012.png", "caption": "..."}]

Usage:
    python scripts/caption.py --data data/curated --mode vlm
"""

import argparse
import json
import re
from pathlib import Path

INSTRUCTION = """Describe this Japanese sumi-e ink painting as one paragraph of 2-4 sentences, for training an image model.
Cover, in this order:
1. Framing: it is a close-up or extreme close-up; say which and what part of the subject fills the frame.
2. Subject: what is painted (plants, animals, landscape elements, objects) and how it is arranged.
3. Technique: brushwork (dry brush, wet wash, splashed ink, fine line), ink tones from deep black to pale grey,
   paper texture, any color accents, how much empty space there is.
Write plain descriptive prose. Do not start with "This image" or "The painting". Do not mention seals, signatures,
or text unless they are clearly visible. Do not guess at artists.
For reference, the prompt that generated it was: "{hint}". Use it only to identify the subject; describe what you see."""

QUALITY_TAGS = {
    "masterpiece", "best quality", "high quality", "highly detailed", "ultra detailed", "8k", "4k", "hd",
    "hdr", "uhd", "absurdres", "highres", "intricate details", "detailed", "award winning", "trending on artstation",
    "sharp focus", "professional", "beautiful", "stunning", "amazing", "high resolution",
}


def clean_prompt(p: str) -> str:
    p = p or ""
    p = re.sub(r"<[^>]+>", " ", p)                   # <lora:...> tags
    p = re.sub(r":\s*-?\d+(\.\d+)?", "", p)          # (word:1.2) weights
    p = re.sub(r"[()\[\]{}]", " ", p)
    p = re.sub(r"\bembedding:\S+", " ", p)
    parts = [t.strip(" .") for t in re.split(r"[,\n|]", p)]
    seen, kept = set(), []
    for t in parts:
        t = re.sub(r"\s+", " ", t)
        low = t.lower()
        if not t or low in QUALITY_TAGS or low in seen:
            continue
        seen.add(low)
        kept.append(t)
    return ", ".join(kept)


def prompt_caption(p: str) -> str:
    body = clean_prompt(p)
    if not body:
        return "Close-up of a Japanese sumi-e ink painting with expressive black ink brushwork on textured paper."
    return f"Close-up of a Japanese sumi-e ink painting: {body}. Black ink brushwork with soft grey washes on textured paper."


def vlm_captions(paths: list[Path], hints: list[str], model_id: str, batch: int, max_new_tokens: int) -> list[str]:
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    proc = AutoProcessor.from_pretrained(model_id, min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28)
    proc.tokenizer.padding_side = "left"

    out: list[str] = []
    for i in range(0, len(paths), batch):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
        texts = []
        for h in hints[i:i + batch]:
            msgs = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": INSTRUCTION.format(hint=clean_prompt(h)[:400])},
            ]}]
            texts.append(proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        inputs = proc(text=texts, images=imgs, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = gen[:, inputs["input_ids"].shape[1]:]
        for t in proc.batch_decode(gen, skip_special_tokens=True):
            out.append(" ".join(t.split()))
        print(f"  captioned {len(out)}/{len(paths)}", end="\r")
    print()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/curated", help="folder with images/ and selection.jsonl")
    ap.add_argument("--mode", choices=["vlm", "prompt"], default="vlm")
    ap.add_argument("--vlm-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    args = ap.parse_args()

    data = Path(args.data)
    with open(data / "selection.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rels = [f"images/{Path(r['file']).name}" for r in rows]
    hints = [r.get("positive_prompt") or "" for r in rows]

    if args.mode == "vlm":
        caps = vlm_captions([data / r for r in rels], hints, args.vlm_model, args.batch, args.max_new_tokens)
    else:
        caps = [prompt_caption(h) for h in hints]

    records = [{"video": r, "caption": c} for r, c in zip(rels, caps)]
    with open(data / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)

    print(f"Wrote {len(records)} captions to {data / 'dataset.json'}. Samples:")
    for rec in records[:5]:
        print(f"  {rec['video']}: {rec['caption']}")


if __name__ == "__main__":
    main()
