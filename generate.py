#!/usr/bin/env python3
"""
Ambigram generator.

Methods:
  img2img  (default) — alternating SD img2img passes; reliable, no gradient issues
  sds                — dual Score Distillation Sampling; experimental

Usage:
  python generate.py --word-a love --word-b hate
  python generate.py --word-a SWIMS                    # self-ambigram
  python generate.py --word-a angel --word-b devil --rounds 50
  python generate.py --word-a love --word-b hate --method sds --steps 1500
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a 180° rotational ambigram.")

    # Words
    p.add_argument("--word-a", required=True, help="Word visible when upright")
    p.add_argument("--word-b", default="", help="Word visible when rotated 180° (omit for self-ambigram)")

    # Method
    p.add_argument("--method", default="img2img", choices=["img2img", "sds"],
                   help="Generation method (default: img2img)")

    # Model
    p.add_argument("--model", default="runwayml/stable-diffusion-v1-5",
                   help="HuggingFace model ID")
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])

    # Image
    p.add_argument("--size", type=int, default=512)

    # img2img-specific
    p.add_argument("--rounds", type=int, default=40,
                   help="[img2img] Number of alternating refinement rounds")
    p.add_argument("--initial-strength", type=float, default=0.65,
                   help="[img2img] Starting img2img noise strength")
    p.add_argument("--final-strength", type=float, default=0.25,
                   help="[img2img] Final img2img noise strength after annealing")
    p.add_argument("--blend-alpha", type=float, default=0.85,
                   help="[img2img] Mix ratio when blending each pass result back in (0–1)")
    p.add_argument("--inference-steps", type=int, default=40,
                   help="[img2img] DDIM steps per denoising pass")

    # SDS-specific
    p.add_argument("--steps", type=int, default=1500, help="[sds] Optimization steps")
    p.add_argument("--lr", type=float, default=5e-3, help="[sds] Adam learning rate")
    p.add_argument("--lambda-b", type=float, default=1.0, help="[sds] Rotated-view loss weight")
    p.add_argument("--min-t", type=int, default=50)
    p.add_argument("--max-t", type=int, default=950)
    p.add_argument("--anneal-max-t", type=int, default=400)

    # Shared prompt
    p.add_argument("--prompt-template",
                   default="the word '{word}' in bold black sans-serif typography on a plain white background, high contrast, sharp edges, perfectly legible",
                   help="Prompt template; {word} is substituted")
    p.add_argument("--neg-prompt",
                   default="blurry, noisy, illegible, low quality, watermark, artistic, decorative, handwriting, multiple words",
                   help="Negative prompt")
    p.add_argument("--guidance-scale", type=float, default=10.0)

    # Output
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--save-every", type=int, default=5,
                   help="Save comparison image every N rounds/steps")
    p.add_argument("--use-wandb", action="store_true")

    # Device
    p.add_argument("--device", default="")

    return p.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    log.warning("No GPU found — this will be very slow on CPU.")
    return torch.device("cpu")


def load_pipeline(model_id: str, dtype_str: str, device: torch.device):
    try:
        from diffusers import StableDiffusionPipeline, DDPMScheduler
    except ImportError:
        log.error("diffusers not installed. Run: pip install diffusers")
        sys.exit(1)

    dtype = torch.float16 if dtype_str == "float16" else torch.float32
    log.info("Loading %s …", model_id)

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    pipe.vae.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.enable_attention_slicing()

    # SDS needs DDPM alphas; img2img uses its own scheduler internally
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    return pipe


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    log.info("Using device: %s", device)

    pipe = load_pipeline(args.model, args.dtype, device)
    output_dir = str(Path(args.output_dir) / f"{args.word_a}__{args.word_b or args.word_a}")

    if args.method == "img2img":
        from src.img2img_generator import AlternatingImg2ImgGenerator, Img2ImgConfig
        cfg = Img2ImgConfig(
            word_a=args.word_a,
            word_b=args.word_b,
            image_size=args.size,
            prompt_template=args.prompt_template,
            neg_prompt=args.neg_prompt,
            num_rounds=args.rounds,
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            initial_strength=args.initial_strength,
            final_strength=args.final_strength,
            blend_alpha=args.blend_alpha,
            output_dir=output_dir,
            save_every=args.save_every,
        )
        AlternatingImg2ImgGenerator(cfg, pipe, device).run()

    else:  # sds
        from src.optimizer import AmbigramOptimizer, OptimizerConfig
        cfg = OptimizerConfig(
            word_a=args.word_a,
            word_b=args.word_b,
            image_size=args.size,
            prompt_template=args.prompt_template,
            neg_prompt=args.neg_prompt,
            num_steps=args.steps,
            lr=args.lr,
            lambda_b=args.lambda_b,
            guidance_scale=args.guidance_scale,
            min_t=args.min_t,
            max_t=args.max_t,
            anneal_max_t=args.anneal_max_t,
            output_dir=output_dir,
            log_every=args.save_every,
            use_wandb=args.use_wandb,
        )
        AmbigramOptimizer(cfg, pipe, device).run()


if __name__ == "__main__":
    main()
