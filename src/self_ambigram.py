from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from tqdm import tqdm

from .losses.clip_loss import CLIPLoss
from .utils.image import blend_init, render_text_image, rotate_180, save_comparison, save_image

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    word: str = "SWIMS"

    # CLIP
    clip_model: str = "ViT-L-14"
    clip_pretrained: str = "openai"
    n_augments: int = 16

    # Per-glyph optimisation
    glyph_size: int = 256
    num_steps: int = 500
    lr: float = 2e-2

    # Regularisation
    lambda_tv: float = 2e-3
    lambda_bw: float = 0.3
    # Output
    output_dir: str = "outputs"
    log_every: int = 100


# ---------------------------------------------------------------------------
# Letter-pair helpers
# ---------------------------------------------------------------------------

def letter_pairs(word: str) -> list[tuple[str, str]]:
    """
    Return the unique letter pairs for a self-ambigram.

    Pair (a, b) at index i means: glyph reads 'a' upright and 'b' rotated 180°.
    For even-length words every letter has a distinct partner.
    For odd-length words the middle letter pairs with itself.

    "SWIMS" → [('S','S'), ('W','M'), ('I','I')]
    "NOON"  → [('N','N'), ('O','O')]
    """
    N = len(word)
    w = word.upper()
    return [(w[i], w[N - 1 - i]) for i in range((N + 1) // 2)]


def compose(glyphs: list[torch.Tensor], N: int) -> torch.Tensor:
    """
    Compose unique glyphs into the full N-character ambigram image.

    Uses the identity:  rotate_180([A | B | … | Z]) = [rot(Z) | … | rot(B) | rot(A)]

    Layout (even N=4):  [g0 | g1 | rot(g1) | rot(g0)]
    Layout (odd  N=5):  [g0 | g1 | g2 | rot(g1) | rot(g0)]

    When the whole composed image is rotated 180°, the identity above gives back
    the same tile sequence — so the word reads identically in both orientations.
    """
    k = len(glyphs)
    tiles: list[torch.Tensor] = list(glyphs)

    # Append the rotated second half, excluding the centre tile for odd words
    start = k - 2 if N % 2 == 1 else k - 1
    for i in range(start, -1, -1):
        tiles.append(rotate_180(glyphs[i]))

    return torch.cat(tiles, dim=-1)  # concat along width (last dim of CHW)


# ---------------------------------------------------------------------------
# Regularisation helpers
# ---------------------------------------------------------------------------

def _total_variation(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0)
    return (
        (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean() +
        (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    )


# ---------------------------------------------------------------------------
# Per-glyph optimiser
# ---------------------------------------------------------------------------

_PROMPTS = [
    "the letter {ch} in bold black sans-serif typography on a plain white background",
    "a single uppercase {ch} character in black ink on white",
    "the capital letter {ch}, high contrast, crisp edges",
]


class GlyphPairOptimizer:
    """
    Optimise a single square glyph that reads as char_a upright
    and char_b when rotated 180°.
    """

    def __init__(
        self,
        char_a: str,
        char_b: str,
        config: Config,
        clip_loss: CLIPLoss,
        device: torch.device,
    ) -> None:
        self.char_a = char_a.upper()
        self.char_b = char_b.upper()
        self.cfg = config
        self.clip_loss = clip_loss
        self.device = device

        self.feat_a = clip_loss.encode_prompts([t.format(ch=self.char_a) for t in _PROMPTS])
        self.feat_b = clip_loss.encode_prompts([t.format(ch=self.char_b) for t in _PROMPTS])

    def run(self, save_dir: Path | None = None) -> torch.Tensor:
        s = self.cfg.glyph_size

        # Single-channel (grayscale) optimisation — colour noise is physically impossible
        # with one channel; no need for a separate grayscale regularisation term.
        img_a = render_text_image(self.char_a, (s, s)).mean(0, keepdim=True).to(self.device)
        img_b = render_text_image(self.char_b, (s, s)).mean(0, keepdim=True).to(self.device)

        image = (img_a + rotate_180(img_b)) * 0.5   # (1, H, W)
        image = image.detach().clone()
        image.requires_grad_(True)

        opt = torch.optim.Adam([image], lr=self.cfg.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.cfg.num_steps, eta_min=self.cfg.lr * 0.1
        )

        label = f"{self.char_a}↔{self.char_b}"
        pbar = tqdm(range(self.cfg.num_steps), desc=f"  glyph {label}", leave=False)
        for step in pbar:
            # Expand to RGB for CLIP; gradient flows back through expand to the 1-ch image
            rgb       = image.expand(3, -1, -1)
            rgb_rot   = rotate_180(image).expand(3, -1, -1)

            loss_a, loss_b = self.clip_loss.forward_pair(
                rgb, rgb_rot, self.feat_a, self.feat_b
            )
            reg = (
                self.cfg.lambda_tv * _total_variation(image) +
                self.cfg.lambda_bw * (image * (1.0 - image)).mean()
            )
            loss = loss_a + loss_b + reg

            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

            with torch.no_grad():
                image.data.clamp_(0.0, 1.0)

            pbar.set_postfix(a=f"{loss_a:.3f}", b=f"{loss_b:.3f}")

            if save_dir and (step % self.cfg.log_every == 0 or step == self.cfg.num_steps - 1):
                save_comparison(
                    image.detach().cpu().expand(3, -1, -1),
                    save_dir / f"step_{step:04d}.png",
                    word_a=self.char_a,
                    word_b=self.char_b,
                )

        return image.detach().cpu().expand(3, -1, -1).clone()


# ---------------------------------------------------------------------------
# Top-level generator
# ---------------------------------------------------------------------------

class SelfAmbigramGenerator:
    """
    Generate a self-ambigram by optimising each letter pair independently,
    then composing the glyphs into the final word image.
    """

    def __init__(self, config: Config, device: torch.device) -> None:
        self.cfg = config
        self.device = device
        self.word = config.word.upper()

        log.info("Loading CLIP %s (%s)…", config.clip_model, config.clip_pretrained)
        self.clip_loss = CLIPLoss(
            model_name=config.clip_model,
            pretrained=config.clip_pretrained,
            device=device,
            n_augments=config.n_augments,
        )

        self.pairs = letter_pairs(self.word)
        N = len(self.word)
        pair_str = ", ".join(
            f"{a}↔{b}" + (" [mid]" if N % 2 == 1 and i == len(self.pairs) - 1 else "")
            for i, (a, b) in enumerate(self.pairs)
        )
        log.info("Word: %s   Pairs: %s", self.word, pair_str)

    def run(self) -> torch.Tensor:
        out = Path(self.cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        glyphs: list[torch.Tensor] = []
        for i, (char_a, char_b) in enumerate(self.pairs):
            log.info("[%d/%d] Optimising glyph %s↔%s", i + 1, len(self.pairs), char_a, char_b)
            glyph_dir = out / f"glyph_{i}_{char_a}_{char_b}"
            glyph_dir.mkdir(exist_ok=True)

            glyph = GlyphPairOptimizer(
                char_a, char_b, self.cfg, self.clip_loss, self.device
            ).run(save_dir=glyph_dir)

            glyphs.append(glyph)
            save_image(glyph, out / f"glyph_{i}_{char_a}_{char_b}.png")
            save_image(rotate_180(glyph), out / f"glyph_{i}_{char_a}_{char_b}_rot.png")

        composed = compose(glyphs, len(self.word))
        save_comparison(composed, out / "final.png", word_a=self.word, word_b=self.word)
        save_image(composed, out / "final_upright.png")
        save_image(rotate_180(composed), out / "final_rotated.png")

        log.info("Done → %s/final.png", out)
        return composed
