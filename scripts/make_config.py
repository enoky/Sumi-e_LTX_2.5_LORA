#!/usr/bin/env python3
"""Write the LTX-2.5 trainer YAML for this image-only, video-branch style LoRA.

Based on packages/ltx-trainer/configs/t2v_lora_low_vram.yaml and v2v_ic_lora.yaml from
Lightricks/LTX-2 (checked against commit f0f2f86, 2026-11-12). Differences from the stock t2v config:
  * no audio block: images have no audio, so only the video branch is trained (video-only target modules)
  * single-frame buckets (images are 1-frame clips)
  * GPU profile picks quantization / optimizer so it fits Colab's A100 40GB, A100 80GB or H100 80GB

Usage:
    python scripts/make_config.py --models /content/models/ltx-2.5 --data /content/data/curated \
        --out /content/configs/sumi_e.yaml --gpu a100-40
"""

import argparse
from pathlib import Path

import yaml

PROFILES = {
    # 22B transformer: ~44 GB in bf16, ~22 GB in int8.
    "a100-40": {"quantization": "int8-quanto", "optimizer_type": "adamw8bit", "te_8bit": True},
    "a100-80": {"quantization": None, "optimizer_type": "adamw", "te_8bit": False},
    "h100": {"quantization": None, "optimizer_type": "adamw", "te_8bit": False},
}

VALIDATION_PROMPTS = [
    "{t} Extreme close-up of a single bamboo stalk and three slender leaves painted in black sumi ink on cream "
    "rice paper. A soft breeze makes the leaves sway gently while the wet ink slowly bleeds into the paper fibers. "
    "Dry-brush texture on the stalk, pale grey washes behind it, wide areas of empty paper. Static camera.",
    "{t} Close-up of a koi fish painted in sumi-e ink, its body formed by a few confident brushstrokes in deep "
    "black and diluted grey. The koi turns slowly through still water suggested by faint horizontal washes, a single "
    "small red accent on its head. Slow camera push-in, visible paper grain.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="folder holding the LTX-2.5 download (hf --local-dir)")
    ap.add_argument("--data", required=True, help="folder holding dataset.json and images/")
    ap.add_argument("--out", required=True, help="where to write the YAML")
    ap.add_argument("--output-dir", default=None, help="training outputs (default: <data>/../outputs/sumi_e_lora)")
    ap.add_argument("--precomputed", default=None, help="latent cache (default: <data>/.precomputed)")
    ap.add_argument("--gpu", choices=sorted(PROFILES), default="a100-40")
    ap.add_argument("--trigger", default="sumi-e ink painting,")
    ap.add_argument("--resolution", type=int, default=1024, help="square bucket side; multiple of 32")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--validation-interval", type=int, default=500, help="0 disables validation videos")
    ap.add_argument("--checkpoint-interval", type=int, default=250)
    ap.add_argument("--transformer", default="diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors")
    ap.add_argument("--text-encoder", default="text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")
    ap.add_argument("--video-vae", default="vae/ltx-2.5-video-vae-conv-bf16.safetensors")
    args = ap.parse_args()

    if args.resolution % 32:
        raise SystemExit("--resolution must be a multiple of 32")

    prof = PROFILES[args.gpu]
    models, data = Path(args.models), Path(args.data)
    output_dir = args.output_dir or str(data.parent / "outputs" / "sumi_e_lora")
    precomputed = args.precomputed or str(data / ".precomputed")

    cfg = {
        "training_strategy": {
            "name": "flexible",
            "model": {
                "model_path": str(models / args.transformer),
                "text_encoder_path": str(models / args.text_encoder),
                "video_vae_path": str(models / args.video_vae),
                "audio_vae_path": None,
                "load_text_encoder_in_8bit": prof["te_8bit"],
            },
            "training_mode": "lora",
            "lora": {
                "rank": args.rank,
                "alpha": args.rank,
                "dropout": 0.0,
                # Video branch only: self-attention, text cross-attention, feed-forward.
                "target_modules": [
                    "attn1.to_k", "attn1.to_q", "attn1.to_v", "attn1.to_out.0",
                    "attn2.to_k", "attn2.to_q", "attn2.to_v", "attn2.to_out.0",
                    "ff.net.0.proj", "ff.net.2",
                ],
            },
            "quantization": prof["quantization"],
            "data": {
                "dataset_file": str(data / "dataset.json"),
                "preprocessed_data_root": precomputed,
                "num_dataloader_workers": 2,
                "resolution_buckets": [[args.resolution, args.resolution, 1]],
                "lora_trigger": args.trigger,
                "skip_audio": True,
            },
            "video": {"is_generated": True, "latents_dir": "latents"},
            "optimizer": {
                "learning_rate": args.lr,
                "optimizer_type": prof["optimizer_type"],
                "scheduler_type": "linear",
                "scheduler_params": {},
                "gradient_checkpointing": True,
            },
            "timestep_sampling": {
                "timestep_sampling_mode": "shifted_logit_normal",
                "timestep_sampling_params": {},
            },
            "validation": {
                "samples": [{"prompt": p.format(t=args.trigger)} for p in VALIDATION_PROMPTS],
                "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted, photo, 3d render",
                "video_dims": [512, 512, 49],
                "frame_rate": 25.0,
                "seed": 42,
                "inference_steps": 25,
                "video_cfg_scale": 3.0,
                "audio_cfg_scale": 7.0,
                "video_stg_scale": 1.0,
                "audio_stg_scale": 1.0,
                "stg_blocks": [28],
                "guidance_rescale": 0.7,
                "video_modality_guidance_scale": 3.0,
                "audio_modality_guidance_scale": 3.0,
                "generate_audio": False,
            },
        },
        "optimization": {
            "steps": args.steps,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "mixed_precision_mode": "bf16",
            "offload_optimizer_during_validation": True,
        },
        "validation": {
            "interval": args.validation_interval or None,
            # Skipping the step-0 baseline saves ~10 min of GPU time; set false to compare against the base model.
            "skip_initial_validation": True,
        },
        "checkpoints": {
            "load_checkpoint": None,
            "interval": args.checkpoint_interval,
            "keep_last_n": -1,
            "precision": "bfloat16",
        },
        "wandb": {"enabled": False, "project": "sumi-e-ltx25", "entity": None, "tags": ["ltx2.5", "lora", "sumi-e"],
                  "log_validation_videos": True},
        "allow_stale_preprocessed": False,
        "seed": 42,
        "output_dir": output_dir,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, width=120, allow_unicode=True)
    print(f"Wrote {args.out} (gpu profile {args.gpu}, quantization={prof['quantization']}, "
          f"bucket {args.resolution}x{args.resolution}x1, {args.steps} steps)")


if __name__ == "__main__":
    main()
