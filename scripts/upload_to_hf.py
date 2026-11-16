#!/usr/bin/env python3
"""Upload a trained LoRA checkpoint to a Hugging Face model repo, with a model card and sample videos.

Needs a token with WRITE access in $HF_TOKEN (or pass --token). The repo is created private unless --public.

Usage:
    python scripts/upload_to_hf.py --repo your-name/sumi-e-ltx-2.5-lora \
        --lora /content/drive/MyDrive/sumi_e_ltx25/outputs/sumi_e_lora \
        --samples /content/drive/MyDrive/sumi_e_ltx25/tests

--lora accepts a weights .safetensors file, a checkpoint step dir, or the training output dir
(picks the latest checkpoint, or the one named with --checkpoint step_01500).
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

CARD = """---
base_model: Lightricks/LTX-2.5
base_model_relation: adapter
library_name: ltx-pipelines
license: other
license_name: ltx-2-community-license
license_link: https://huggingface.co/Lightricks/LTX-2.5
datasets:
  - takara-ai/Sumi-e_no_kurozuappu_ink_painting_closeups
tags:
  - lora
  - text-to-video
  - ltx-video
  - ltx-2.5
  - sumi-e
  - ink-painting
  - style
pipeline_tag: text-to-video
---

# Sumi-e ink painting LoRA for LTX-2.5

A style LoRA that renders LTX-2.5 videos as Japanese sumi-e ink paintings: black ink brushwork, grey washes,
paper texture and plenty of empty space.

**Trigger:** start your prompt with `{trigger}`

## Files

- `{weights_name}`: LoRA weights ({checkpoint}, rank {rank}), ComfyUI/ltx-pipelines key format
{samples_section}
## Usage

Trained on the **dev** transformer `ltx-2.5-22b-dev-transformer-bf16.safetensors`. With
[ltx-pipelines](https://github.com/Lightricks/LTX-2):

```bash
uv run python -m ltx_pipelines.ti2vid_one_stage \\
    --transformer-path models/ltx-2.5/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors \\
    --text-encoder-path models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \\
    --video-vae-path models/ltx-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors \\
    --audio-vae-path models/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors \\
    --lora {weights_name} 1.0 \\
    --prompt "{trigger} Close-up of bamboo leaves swaying in the wind, wet ink bleeding into rice paper." \\
    --output-path out.mp4
```

Strength 0.7–1.0 works as a starting range. Describing the framing helps: the training images are all close-ups.

## Training

- Data: a curated, deduplicated subset of
  [takara-ai/Sumi-e_no_kurozuappu_ink_painting_closeups](https://huggingface.co/datasets/takara-ai/Sumi-e_no_kurozuappu_ink_painting_closeups)
  (synthetic images), recaptioned in natural language
- Images trained as single-frame clips, so the LoRA learns the look; motion comes from the base model
- Video branch only (self-attention, text cross-attention, feed-forward), Lightricks `ltx-trainer`

## Limitations

- The training images are AI-generated, so the LoRA picks up that generator's flaws along with the style.
- Strong bias toward close-up framing.

## License and credit

The LoRA is a derivative of LTX-2.5 and is subject to the
[LTX-2.x Community License](https://huggingface.co/Lightricks/LTX-2.5).
Training data: Takara.ai, *Sumi-e no Kurozuappu Ink Painting Closeups* (2024), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
"""


def resolve(lora: Path, name: str | None) -> tuple[str, Path]:
    if lora.is_file():
        return (lora.parent.name if lora.name == "weights.safetensors" else lora.stem), lora
    if (lora / "weights.safetensors").is_file():
        return lora.name, lora / "weights.safetensors"
    root = lora / "checkpoints" if (lora / "checkpoints").is_dir() else lora
    steps = sorted(p for p in root.glob("step_*") if (p / "weights.safetensors").is_file())
    if not steps:
        raise SystemExit(f"No checkpoints found under {lora}")
    if name:
        match = [p for p in steps if p.name == name]
        if not match:
            raise SystemExit(f"{name} not found. Available: {[p.name for p in steps]}")
        return name, match[0] / "weights.safetensors"
    return steps[-1].name, steps[-1] / "weights.safetensors"


def lora_rank(path: Path) -> str:
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as f:
            for k in f.keys():
                if "lora_A" in k or "lora_down" in k:
                    return str(f.get_slice(k).get_shape()[0])
    except Exception:  # noqa: BLE001
        pass
    return "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. your-username/sumi-e-ltx-2.5-lora")
    ap.add_argument("--lora", required=True)
    ap.add_argument("--checkpoint", default=None, help="step dir name, e.g. step_01500 (default: latest)")
    ap.add_argument("--samples", default=None, help="folder of test MP4s (from test_lora.py) to include")
    ap.add_argument("--max-samples", type=int, default=6)
    ap.add_argument("--trigger", default="sumi-e ink painting,")
    ap.add_argument("--weights-name", default="sumi-e_ltx-2.5_lora.safetensors")
    ap.add_argument("--public", action="store_true", help="create the repo public (default: private)")
    ap.add_argument("--token", default=None, help="write token (default: $HF_TOKEN)")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    token = args.token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    user = api.whoami()["name"]
    checkpoint, weights = resolve(Path(args.lora), args.checkpoint)
    print(f"Uploading {weights} ({checkpoint}) as {args.repo} (logged in as {user})")

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        shutil.copy2(weights, stage / args.weights_name)

        samples_section = ""
        if args.samples:
            # Prefer LoRA renders (skip the base-model column) for the card.
            vids = [v for v in sorted(Path(args.samples).glob("*.mp4")) if not v.stem.endswith("__base")]
            vids = vids[: args.max_samples]
            if vids:
                (stage / "samples").mkdir()
                lines = []
                for v in vids:
                    shutil.copy2(v, stage / "samples" / v.name)
                    lines.append(f'- <video src="samples/{v.name}" controls loop muted width="384"></video> `{v.name}`')
                samples_section = "- `samples/`: example renders\n\n## Samples\n\n" + "\n".join(lines) + "\n"

        (stage / "README.md").write_text(CARD.format(
            trigger=args.trigger, weights_name=args.weights_name, checkpoint=checkpoint,
            rank=lora_rank(weights), samples_section=samples_section), encoding="utf-8")

        api.create_repo(args.repo, repo_type="model", private=not args.public, exist_ok=True)
        api.upload_folder(repo_id=args.repo, folder_path=str(stage), repo_type="model",
                          commit_message=f"Upload sumi-e LoRA ({checkpoint})")

    print(f"Done: https://huggingface.co/{args.repo} ({'public' if args.public else 'private'})")


if __name__ == "__main__":
    main()
