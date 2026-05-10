from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def rotate_180(x: torch.Tensor) -> torch.Tensor:
    """Rotate a NCHW or CHW tensor 180 degrees (flip both spatial dims)."""
    return x.flip([-2, -1])


def render_text_image(
    word: str,
    size: tuple[int, int] = (512, 512),
    bg_color: int = 255,
    fg_color: int = 0,
    padding: float = 0.15,
) -> torch.Tensor:
    """Render a word centered on a white background. Returns CHW float32 in [0, 1]."""
    h, w = size
    img = Image.new("RGB", (w, h), color=(bg_color, bg_color, bg_color))
    draw = ImageDraw.Draw(img)

    # Try system fonts in order of preference, fall back to default
    font = None
    for name in ("Arial", "DejaVuSans", "LiberationSans", "FreeSans"):
        for ext in (".ttf", ".TTF"):
            for search in ("/usr/share/fonts", "/Library/Fonts", "/System/Library/Fonts"):
                path = Path(search)
                matches = list(path.rglob(f"*{name}*{ext}"))
                if matches:
                    try:
                        target_h = int(h * (1 - 2 * padding))
                        font = ImageFont.truetype(str(matches[0]), size=target_h)
                        break
                    except Exception:
                        continue
            if font:
                break
        if font:
            break

    if font is None:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), word, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (w - text_w) // 2 - bbox[0]
    y = (h - text_h) // 2 - bbox[1]
    draw.text((x, y), word, font=font, fill=(fg_color, fg_color, fg_color))

    tensor = torch.from_numpy(np.array(img)).float() / 255.0
    return tensor.permute(2, 0, 1)  # CHW


def blend_init(
    word_a: str,
    word_b: str,
    size: tuple[int, int] = (512, 512),
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Blend rendered word_a (upright) with rendered word_b (rotated 180°).
    Returns a CHW tensor in [0, 1] — a good starting point for optimization.
    """
    img_a = render_text_image(word_a, size)
    img_b = render_text_image(word_b, size)
    img_b_rot = rotate_180(img_b)
    return alpha * img_a + (1 - alpha) * img_b_rot


def save_comparison(
    image: torch.Tensor,
    path: str | Path,
    word_a: str = "",
    word_b: str = "",
) -> None:
    """Save a side-by-side image: upright | rotated 180°."""
    img = image.detach().cpu().clamp(0, 1)
    if img.dim() == 4:
        img = img.squeeze(0)

    h, w = img.shape[-2], img.shape[-1]
    gap = 8

    canvas = torch.ones(3, h, w * 2 + gap)
    canvas[:, :, :w] = img
    canvas[:, :, w + gap:] = rotate_180(img)

    pil = Image.fromarray((canvas.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

    if word_a or word_b:
        draw = ImageDraw.Draw(pil)
        draw.text((4, 4), word_a, fill=(180, 0, 0))
        draw.text((w + gap + 4, 4), f"{word_b} (rotated)", fill=(0, 0, 180))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pil.save(path)


def save_image(image: torch.Tensor, path: str | Path) -> None:
    img = image.detach().cpu().clamp(0, 1)
    if img.dim() == 4:
        img = img.squeeze(0)
    pil = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pil.save(path)


def logit(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))
