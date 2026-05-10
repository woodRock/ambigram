#!/usr/bin/env python3
"""
Ambigram generator via dual SDS optimization.

Usage:
  python generate.py --word-a love --word-b hate
  python generate.py --word-a SWIMS                  # self-ambigram
  python generate.py --word-a angel --word-b devil --steps 2000 --model stabilityai/stable-diffusion-2-1
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
    p = argparse.ArgumentParser(
        description="Generate a rotational ambigram via dual SDS optimization."
    )
    # Words
    p.add_argument("--word-a", required=True, help="Word visible when upright")
    p.add_argument("--word-b", default="", help="Word visible when rotated 180° (omit for self-ambigram)")

    # Model
    p.add_argument("--model", default="runwayml/stable-diffusion-v1-5",
                   help="HuggingFace model ID (SD 1.5 recommended)")
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"],
                   help="Dtype for the diffusion model weights")

    # Image
    p.add_argument("--size", type=int, default=512, help="Output image size (square)")

    # Optimization
    p.add_argument("--steps", type=int, default=1500, help="Optimization steps")
    p.add_argument("--lr", type=float, default=5e-3, help="Adam learning rate")
    p.add_argument("--lambda-b", type=float, default=1.0,
                   help="Weight for the rotated-view SDS loss (1.0 = balanced)")

    # SDS
    p.add_argument("--guidance-scale", type=float, default=7.5,
                   help="Classifier-free guidance scale")
    p.add_argument("--min-t", type=int, default=50,
                   help="Minimum diffusion timestep sampled during SDS")
    p.add_argument("--max-t", type=int, default=950,
                   help="Initial maximum diffusion timestep")
    p.add_argument("--anneal-max-t", type=int, default=400,
                   help="Target max-t after annealing (lower = finer detail focus)")

    # Prompt
    p.add_argument("--prompt-template", default="the word '{word}' in clear bold typography on white background",
                   help="Prompt template; {word} is substituted")
    p.add_argument("--neg-prompt", default="blurry, noisy, illegible, low quality, watermark",
                   help="Negative prompt")

    # Output
    p.add_argument("--output-dir", default="outputs", help="Directory for saved images")
    p.add_argument("--log-every", type=int, default=100, help="Save comparison image every N steps")
    p.add_argument("--use-wandb", action="store_true", help="Log losses to Weights & Biases")

    # Device
    p.add_argument("--device", default="", help="cuda / cpu / mps (auto-detected if empty)")

    return p.parse_args()


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

    # Replace scheduler with DDPM for correct alphas_cumprod
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Freeze everything — we only optimise the image
    pipe.vae.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    # Enable attention slicing to save VRAM when not on H100
    pipe.enable_attention_slicing()

    return pipe


def main() -> None:
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        log.warning("No GPU found — optimization will be very slow on CPU.")
    log.info("Using device: %s", device)

    # ── Pipeline ──────────────────────────────────────────────────────────
    pipe = load_pipeline(args.model, args.dtype, device)

    # ── Config ────────────────────────────────────────────────────────────
    from src.optimizer import AmbigramOptimizer, OptimizerConfig

    output_dir = str(Path(args.output_dir) / f"{args.word_a}__{args.word_b or args.word_a}")
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
        log_every=args.log_every,
        use_wandb=args.use_wandb,
    )

    # ── Run ───────────────────────────────────────────────────────────────
    optimizer = AmbigramOptimizer(cfg, pipe, device)
    optimizer.run()


if __name__ == "__main__":
    main()
