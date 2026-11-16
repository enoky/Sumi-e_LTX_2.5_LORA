# Sumi-e LTX-2.5 LoRA

A style LoRA for [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5), trained on
[takara-ai/Sumi-e_no_kurozuappu_ink_painting_closeups](https://huggingface.co/datasets/takara-ai/Sumi-e_no_kurozuappu_ink_painting_closeups)
(1,250 synthetic Japanese ink-painting close-ups, CC BY 4.0) with Lightricks' official
[`ltx-trainer`](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-trainer).

## Quick start (Colab)

Open [`colab/train_sumi_e_lora.ipynb`](colab/train_sumi_e_lora.ipynb) in Colab and run the cells in order. The
notebook explains the three setup steps (A100/H100 runtime, accept the LTX-2.5 terms, `HF_TOKEN` secret).

## What the pipeline does

| Step | Script | Notes |
|------|--------|-------|
| Download | `scripts/download_dataset.py` | Saves PNGs plus `metadata.jsonl` (seed, SD prompt, generator model...), and prints the image sizes. |
| Curate | `scripts/curate.py` | CLIP near-duplicate removal, then k-means to pick ~300 diverse images. Writes contact sheets; put rejects in `exclude.txt` and rerun. |
| Caption | `scripts/caption.py` | `--mode vlm`: Qwen2.5-VL writes prose captions (framing, subject, brushwork). `--mode prompt`: cleans the SD prompts, no GPU needed. Writes `dataset.json`. |
| Config | `scripts/make_config.py` | Trainer YAML: single-frame 1024² bucket, video-branch-only LoRA (rank 32), no audio, trigger `sumi-e ink painting,`. Picks int8 + 8-bit AdamW on A100 40GB, bf16 on 80GB cards. |
| Train | LTX-2 `scripts/preprocess.py` then `scripts/train.py` | Uses the **dev** transformer. Writes validation videos every 500 steps and checkpoints every 250 steps. |

## Design notes

- **Images train the look, not the motion.** Each image is a one-frame clip, so the LoRA learns ink texture, tone and
  composition, and LTX supplies the motion. For ink-specific motion (bleeding, strokes appearing), add short clips.
- **Synthetic source.** The images come from an SD-family model, so the LoRA inherits that model's flaws. That's why
  curation and reviewing the contact sheets matter.
- **Captions** state the framing ("close-up", "extreme close-up"). Leaving those words out of a prompt then lets you
  get wider shots.
- Written against LTX-2 commit `f0f2f86` (2026-11-12); the notebook pins it.

## Troubleshooting

- **`CUDA ok: False` after `uv sync`:** the trainer pins a CUDA 13 build of PyTorch. If Colab's driver is older (see
  `nvidia-smi`), the GPU won't be visible to the trainer environment. Try an H100 runtime, which usually has a newer
  driver.
- **Out of memory on A100 40GB:** lower `--resolution` to 768, or `--rank` to 16.
- **401/403 when downloading models:** accept the terms on the LTX-2.5 model page and check that `HF_TOKEN` is a Read token.

## License and credit

Dataset: Takara.ai, *Sumi-e no Kurozuappu Ink Painting Closeups* (2024), CC BY 4.0. Credit it when you share the LoRA.
The LTX-2.5 weights are under the LTX-2.x Community License. The trainer code is Apache 2.0.
