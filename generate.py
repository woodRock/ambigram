#!/usr/bin/env python3
"""
Self-ambigram generator — CLIP-guided per-glyph optimisation.

Splits the word into letter pairs, optimises each glyph simultaneously,
then composes them into a word that reads the same rotated 180°.

Usage:
  python generate.py --word SWIMS
  python generate.py --word NOON --steps 800 --glyph-size 384
  python generate.py --word SWIMS --mode bezier --n-strokes 12
  python generate.py --word SWIMS --use-perceptual --use-classifier
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

    # CLIP
    p.add_argument("--clip-model",      default="ViT-L-14")
    p.add_argument("--clip-pretrained", default="openai")
    p.add_argument("--n-augments",      type=int, default=16)
    p.add_argument("--torch-compile",   action="store_true",
                   help="Apply torch.compile to the CLIP model (CUDA only)")

    # Representation
    p.add_argument("--mode", choices=["pixel", "bezier"], default="pixel",
                   help="Glyph representation: direct pixels or Bézier strokes")
    p.add_argument("--n-strokes",    type=int,   default=4,
                   help="Number of Bézier strokes per glyph (bezier mode only)")
    p.add_argument("--stroke-width", type=float, default=0.04,
                   help="Stroke half-width as a fraction of canvas size (bezier mode)")

    # Optimisation
    p.add_argument("--glyph-size", type=int,   default=256)
    p.add_argument("--steps",      type=int,   default=500)
    p.add_argument("--lr",         type=float, default=2e-2)

    # Pixel regularisation
    p.add_argument("--lambda-tv",     type=float, default=5e-3)
    p.add_argument("--lambda-bw",     type=float, default=0.8)
    p.add_argument("--lambda-anchor", type=float, default=0.5,
                   help="Anchor loss weight: pulls image toward blended reference "
                        "to suppress background noise (pixel mode)")

    # Character classifier
    p.add_argument("--use-classifier",  action="store_true",
                   help="Add character-classifier readability loss")
    p.add_argument("--classifier-path", default="data/char_classifier.pth")
    p.add_argument("--lambda-char",     type=float, default=0.5)

    # Perceptual loss
    p.add_argument("--use-perceptual", action="store_true",
                   help="Add VGG perceptual loss towards rendered reference letters")
    p.add_argument("--lambda-perc",    type=float, default=0.1)

    # Output
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
        torch_compile=args.torch_compile,
        mode=args.mode,
        n_strokes=args.n_strokes,
        stroke_width=args.stroke_width,
        glyph_size=args.glyph_size,
        num_steps=args.steps,
        lr=args.lr,
        lambda_tv=args.lambda_tv,
        lambda_bw=args.lambda_bw,
        lambda_anchor=args.lambda_anchor,
        use_classifier=args.use_classifier,
        classifier_path=args.classifier_path,
        lambda_char=args.lambda_char,
        use_perceptual=args.use_perceptual,
        lambda_perc=args.lambda_perc,
        output_dir=str(Path(args.output_dir) / args.word.upper()),
        log_every=args.log_every,
    )

    SelfAmbigramGenerator(cfg, device).run()


if __name__ == "__main__":
    main()
