#!/usr/bin/env python3
"""
Self-ambigram generator — CLIP-guided per-glyph optimisation.

Splits the word into letter pairs, optimises each glyph independently,
then composes them into a word that reads the same rotated 180°.

Usage:
  python generate.py --word SWIMS
  python generate.py --word NOON --steps 800 --glyph-size 384
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a self-ambigram via CLIP-guided per-glyph optimisation."
    )
    p.add_argument("--word", required=True, help="Word to ambigramise (e.g. SWIMS, NOON)")

    p.add_argument("--clip-model",      default="ViT-L-14")
    p.add_argument("--clip-pretrained", default="openai")
    p.add_argument("--n-augments",      type=int,   default=16)

    p.add_argument("--glyph-size", type=int,   default=256,  help="Square glyph canvas size in px")
    p.add_argument("--steps",      type=int,   default=500,  help="Optimisation steps per glyph")
    p.add_argument("--lr",         type=float, default=2e-2)

    p.add_argument("--lambda-tv", type=float, default=2e-3)
    p.add_argument("--lambda-bw", type=float, default=0.3)

    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--log-every",  type=int, default=100)
    p.add_argument("--device",     default="")
    return p.parse_args()


def resolve_device(req: str) -> torch.device:
    if req:
        return torch.device(req)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    log.warning("No GPU found — will be slow on CPU.")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    log.info("Device: %s", device)

    from src.self_ambigram import Config, SelfAmbigramGenerator

    cfg = Config(
        word=args.word,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        n_augments=args.n_augments,
        glyph_size=args.glyph_size,
        num_steps=args.steps,
        lr=args.lr,
        lambda_tv=args.lambda_tv,
        lambda_bw=args.lambda_bw,
        output_dir=str(Path(args.output_dir) / args.word.upper()),
        log_every=args.log_every,
    )

    SelfAmbigramGenerator(cfg, device).run()


if __name__ == "__main__":
    main()
