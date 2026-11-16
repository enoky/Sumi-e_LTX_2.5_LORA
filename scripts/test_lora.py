#!/usr/bin/env python3
"""Generate test videos with the trained LoRA and compare them against the base model.

Runs LTX-2's TI2VidOneStagePipeline on the same dev transformer the LoRA was trained on. Each "variant" is a
(checkpoint, strength) pair; strength 0 means the base model without the LoRA. The model is loaded once per variant
and every prompt is rendered with the same seed, so the columns of the comparison page differ only in the LoRA.

Must run inside the LTX-2 uv environment (it imports ltx_pipelines / ltx_core):

    cd /content/LTX-2 && uv run python /content/sumi/scripts/test_lora.py \
        --models /content/models/ltx-2.5 \
        --lora /content/drive/MyDrive/sumi_e_ltx25/outputs/sumi_e_lora \
        --strengths 0 0.8 1.0 --out /content/drive/MyDrive/sumi_e_ltx25/tests

--lora accepts a weights .safetensors file, one checkpoint step directory, or the training output directory
(uses the latest checkpoint, or the ones named with --checkpoints step_01000 step_02000).

Output: one MP4 per (prompt, variant) plus index.html, a side-by-side grid (rows = prompts, columns = variants).
"""

import argparse
import gc
import html
import json
import re
import time
from dataclasses import replace
from pathlib import Path

import torch

DEFAULT_PROMPTS = [
    # In-distribution: close-ups like the training data.
    "{t} Extreme close-up of bamboo leaves painted in black sumi ink on cream rice paper. The leaves sway gently in a "
    "breeze while wet ink slowly bleeds into the paper fibers. Dry-brush texture, pale grey washes, wide empty paper.",
    "{t} Close-up of a crane's head and long neck drawn with a few confident brushstrokes, a single red accent on its "
    "crown. The crane slowly turns its head toward the camera. Diluted grey ink washes suggest mist behind it.",
    # Motion test: ink-like movement the images can't have taught directly.
    "{t} Close-up of a plum blossom branch. Black ink strokes appear one after another as if painted by an invisible "
    "brush, then pink-grey petals bloom along the branch. Visible paper grain, static camera.",
    # Generalization: wider framing and subjects absent from the dataset.
    "{t} Wide shot of a fishing boat on a misty river at dawn, mountains fading into pale grey washes in the "
    "background. The boat drifts slowly from left to right; ripples are drawn as thin dry-brush lines.",
    "{t} Medium shot of a cat sitting on a wooden veranda, its fur rendered in soft wet ink washes and a few sharp "
    "strokes for whiskers. The cat flicks its tail and blinks. Slow camera push-in.",
    # Leak test: no trigger word. With a well-behaved LoRA this should stay mostly photographic.
    "A red sports car drives along a coastal highway at sunset, the camera tracking alongside it.",
]


def resolve_checkpoints(lora: Path, names: list[str] | None) -> list[tuple[str, Path]]:
    """Return [(label, weights.safetensors)] for the given path."""
    if lora.is_file():
        return [(lora.parent.name if lora.name == "weights.safetensors" else lora.stem, lora)]
    if (lora / "weights.safetensors").is_file():
        return [(lora.name, lora / "weights.safetensors")]
    ckpt_root = lora / "checkpoints" if (lora / "checkpoints").is_dir() else lora
    steps = sorted(p for p in ckpt_root.glob("step_*") if (p / "weights.safetensors").is_file())
    if not steps:
        # Fall back to loose .safetensors files (e.g. older trainer layouts or a renamed export).
        files = sorted(ckpt_root.glob("*.safetensors"))
        if not files:
            raise SystemExit(f"No LoRA weights found under {lora}")
        steps_files = [(f.stem, f) for f in files]
        return steps_files if names is None else [s for s in steps_files if s[0] in names]
    if names:
        chosen = [p for p in steps if p.name in names]
        missing = set(names) - {p.name for p in chosen}
        if missing:
            raise SystemExit(f"Checkpoints not found: {sorted(missing)}. Available: {[p.name for p in steps]}")
        steps = chosen
    else:
        steps = steps[-1:]
    return [(p.name, p / "weights.safetensors") for p in steps]


def slug(s: str, n: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:n]


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="LTX-2.5 download folder (hf --local-dir)")
    ap.add_argument("--lora", required=True, help="weights file, checkpoint step dir, or training output dir")
    ap.add_argument("--checkpoints", nargs="*", default=None, help="step dir names to compare, e.g. step_01000")
    ap.add_argument("--strengths", nargs="+", type=float, default=[0.0, 1.0], help="0 = base model, no LoRA")
    ap.add_argument("--prompts", default=None, help="text file, one prompt per line; {t} is replaced by the trigger")
    ap.add_argument("--trigger", default="sumi-e ink painting,")
    ap.add_argument("--out", default="tests")
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--frames", type=int, default=97, help="8k+1; 97 frames = ~4 s at 24 fps")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--negative-prompt", default=None, help="default: LTX's built-in negative prompt")
    ap.add_argument("--offload", choices=["auto", "none", "cpu", "disk"], default="auto",
                    help="auto: none if the GPU has >= 70 GB, else cpu (needs ~45 GB system RAM)")
    ap.add_argument("--quantization", default=None, help="e.g. fp8-cast (Hopper/Ada GPUs only, not A100)")
    ap.add_argument("--keep-audio", action="store_true", help="keep the model's generated audio track")
    ap.add_argument("--transformer", default="diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors")
    ap.add_argument("--text-encoder", default="text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")
    ap.add_argument("--video-vae", default="vae/ltx-2.5-video-vae-conv-bf16.safetensors")
    ap.add_argument("--audio-vae", default="vae/ltx-2.5-audio-vae-bf16.safetensors")
    args = ap.parse_args()

    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
    from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
    from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, detect_params
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_pipelines.utils.model_paths import ModelPaths
    from ltx_pipelines.utils.quantization_factory import QuantizationKind
    from ltx_pipelines.utils.types import OffloadMode

    if args.width % 32 or args.height % 32:
        raise SystemExit("--width and --height must be multiples of 32")
    if (args.frames - 1) % 8:
        raise SystemExit("--frames must be 8k+1 (e.g. 49, 97, 121)")

    models = Path(args.models)
    paths = {k: models / v for k, v in {
        "transformer": args.transformer, "text_encoder": args.text_encoder,
        "video_vae": args.video_vae, "audio_vae": args.audio_vae}.items()}
    for k, p in paths.items():
        if not p.is_file():
            raise SystemExit(f"Missing {k}: {p}")
    model_paths = ModelPaths.from_split(
        transformer_path=str(paths["transformer"]), text_encoder_path=str(paths["text_encoder"]),
        video_vae_path=str(paths["video_vae"]), audio_vae_path=str(paths["audio_vae"]))

    if args.offload == "auto":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
        offload = OffloadMode.NONE if vram_gb >= 70 else OffloadMode.CPU
        print(f"GPU has {vram_gb:.0f} GB -> offload {offload.name}")
    else:
        offload = OffloadMode[args.offload.upper()]
    quant = QuantizationKind(args.quantization).to_policy(str(paths["transformer"])) if args.quantization else None

    if args.prompts:
        prompts = [line.strip() for line in open(args.prompts, encoding="utf-8") if line.strip()]
    else:
        prompts = DEFAULT_PROMPTS
    prompts = [p.replace("{t}", args.trigger).strip() for p in prompts]

    checkpoints = resolve_checkpoints(Path(args.lora), args.checkpoints)
    variants: list[tuple[str, Path | None, float]] = []
    if 0.0 in args.strengths:
        variants.append(("base", None, 0.0))
    for label, path in checkpoints:
        for s in args.strengths:
            if s != 0.0:
                variants.append((f"{label} @ {s:g}", path, s))
    print("Variants:", [v[0] for v in variants])

    params = detect_params(str(paths["transformer"]))
    neg = args.negative_prompt or DEFAULT_NEGATIVE_PROMPT
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results: dict[tuple[int, str], str] = {}

    for vlabel, lora_path, strength in variants:
        loras = [] if lora_path is None else [
            LoraPathStrengthAndSDOps(str(lora_path), strength, LTXV_LORA_COMFY_RENAMING_MAP)]
        t0 = time.time()
        pipe = TI2VidOneStagePipeline(model_paths=model_paths, loras=loras, quantization=quant, offload_mode=offload)
        print(f"\n[{vlabel}] pipeline ready in {time.time() - t0:.0f}s")
        for i, prompt in enumerate(prompts):
            fname = f"p{i:02d}_{slug(prompt.replace(args.trigger, ''))}__{slug(vlabel, 30)}.mp4"
            t1 = time.time()
            with torch.inference_mode():
                res = pipe(
                    prompt=prompt, negative_prompt=neg, seed=args.seed,
                    height=args.height, width=args.width, num_frames=args.frames, frame_rate=args.fps,
                    num_inference_steps=args.steps,
                    video_guider_params=replace(params.video_guider_params),
                    audio_guider_params=replace(params.audio_guider_params),
                    images=[], tiling_config=AUTO_TILING,
                )
                encode_video(
                    video=res.video, fps=args.fps, audio=res.audio if args.keep_audio else None,
                    output_path=str(out / fname),
                    video_chunks_number=get_video_chunks_number(res.num_frames, res.tiling_config),
                )
            results[(i, vlabel)] = fname
            print(f"  prompt {i}: {fname} ({time.time() - t1:.0f}s)")
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    write_index(out, prompts, [v[0] for v in variants], results, args)
    print(f"\nDone. Open {out / 'index.html'} (or the MP4s) to compare.")


def write_index(out: Path, prompts: list[str], labels: list[str], results: dict, args: argparse.Namespace) -> None:
    head = "".join(f"<th>{html.escape(lbl)}</th>" for lbl in labels)
    rows = []
    for i, p in enumerate(prompts):
        cells = "".join(
            f'<td><video src="{html.escape(results[(i, lbl)])}" controls loop muted autoplay playsinline></video></td>'
            if (i, lbl) in results else "<td>-</td>" for lbl in labels)
        rows.append(f'<tr><td class="p">{html.escape(p)}</td>{cells}</tr>')
    settings = html.escape(json.dumps({k: v for k, v in vars(args).items() if k not in {"prompts"}}, default=str))
    (out / "index.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>LoRA test</title>
<style>body{{font:14px system-ui;background:#f4f1ea;color:#222;margin:16px}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:6px;vertical-align:top}}video{{width:320px;display:block}}
td.p{{max-width:260px;font-size:12px}}code{{font-size:11px;word-break:break-all}}</style>
<h1>Sumi-e LoRA test</h1><p><code>{settings}</code></p>
<table><tr><th>prompt</th>{head}</tr>{''.join(rows)}</table>""", encoding="utf-8")


if __name__ == "__main__":
    main()
