#!/usr/bin/env python3
"""
Generate a synthetic font dataset for training the character classifier.

Each uppercase letter (A-Z) gets `--samples-per-char` augmented renders.
Images are saved as <output-dir>/<LETTER>/<index>.png (26 subdirs, one per class).

Usage:
  python tools/generate_font_dataset.py
  python tools/generate_font_dataset.py --output-dir data/char_dataset --size 64 --samples-per-char 500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.image import render_text_image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir",       default="data/char_dataset")
    p.add_argument("--size",             type=int, default=64)
    p.add_argument("--samples-per-char", type=int, default=300)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)

    augment = T.Compose([
        T.RandomAffine(
            degrees=12,
            translate=(0.12, 0.12),
            scale=(0.75, 1.15),
            interpolation=T.InterpolationMode.BILINEAR,
        ),
        T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.4),
        T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.3),
    ])

    total = 0
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        char_dir = out / ch
        char_dir.mkdir(parents=True, exist_ok=True)

        # Base render at 2× size for quality, then resize
        base = render_text_image(ch, (args.size * 2, args.size * 2))

        for i in range(args.samples_per_char):
            aug = augment(base)                           # (3, H, W)
            # Downscale to target size
            aug = T.functional.resize(aug, [args.size, args.size],
                                      interpolation=T.InterpolationMode.BILINEAR,
                                      antialias=True)
            arr = (aug.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(arr).save(char_dir / f"{i:04d}.png")
            total += 1

    print(f"Saved {total} images to {out}  ({args.samples_per_char} per class)")


if __name__ == "__main__":
    main()
