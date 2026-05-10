#!/usr/bin/env python3
"""
Ambigram generator — CLIP-guided pixel optimisation.

Directly optimises a pixel image to maximise CLIP similarity for word_a (upright)
and word_b (rotated 180°) simultaneously. No diffusion model required.

Usage:
  python generate.py --word-a love --word-b hate
  python generate.py --word-a SWIMS               # self-ambigram
  python generate.py --word-a angel --word-b devil --steps 3000 --lr 1e-2
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
    p = argparse.ArgumentParser(description="Generate a 180° rotational ambigram via CLIP optimisation.")

    p.add_argument("--word-a", required=True, help="Word visible when upright")
    p.add_argument("--word-b", default="", help="Word visible when rotated 180° (omit for self-ambigram)")

    p.add_argument("--clip-model",      default="ViT-L-14",  help="OpenCLIP model name")
    p.add_argument("--clip-pretrained", default="openai",    help="OpenCLIP pretrained weights tag")
    p.add_argument("--n-augments",      type=int, default=32, help="Crops per CLIP loss evaluation")

    p.add_argument("--size",   type=int,   default=512)
    p.add_argument("--steps",  type=int,   default=2000)
    p.add_argument("--lr",     type=float, default=2e-2)

    p.add_argument("--lambda-b",     type=float, default=1.0,  help="Weight of rotated-view loss")
    p.add_argument("--lambda-tv",    type=float, default=2e-3, help="Total variation weight (smoothness)")
    p.add_argument("--lambda-bw",    type=float, default=0.3,  help="Black/white push weight")
    p.add_argument("--lambda-color", type=float, default=2.0,  help="Grayscale constraint weight (eliminates colour noise)")

    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--log-every",  type=int, default=100)
    p.add_argument("--device",     default="")

    return p.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    log.warning("No GPU found — optimisation will be slow on CPU.")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    log.info("Device: %s", device)

    from src.clip_optimizer import CLIPAmbigramOptimizer, CLIPOptimizerConfig

    output_dir = str(Path(args.output_dir) / f"{args.word_a}__{args.word_b or args.word_a}")
    cfg = CLIPOptimizerConfig(
        word_a=args.word_a,
        word_b=args.word_b,
        image_size=args.size,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        n_augments=args.n_augments,
        num_steps=args.steps,
        lr=args.lr,
        lambda_b=args.lambda_b,
        lambda_tv=args.lambda_tv,
        lambda_bw=args.lambda_bw,
        lambda_color=args.lambda_color,
        output_dir=output_dir,
        log_every=args.log_every,
    )

    CLIPAmbigramOptimizer(cfg, device).run()


if __name__ == "__main__":
    main()
