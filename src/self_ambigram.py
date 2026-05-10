from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .bezier_glyph import BezierGlyph
from .losses.clip_loss import CLIPLoss
from .utils.image import render_text_image, rotate_180, save_comparison, save_image

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
    torch_compile: bool = False

    # Representation mode: "pixel" or "bezier"
    mode: str = "pixel"
    n_strokes: int = 4            # bezier only
    stroke_width: float = 0.035   # bezier only, in [0, 1] canvas fraction

    # Per-glyph optimisation
    glyph_size: int = 256
    num_steps: int = 500
    lr: float = 2e-2

    # Pixel regularisation
    lambda_tv: float = 5e-3      # total variation — suppresses fine noise
    lambda_bw: float = 0.8       # black-and-white push — keeps strokes crisp
    lambda_anchor: float = 0.5   # pulls image toward blended reference — prevents bg noise

    # Character classifier
    use_classifier: bool = False
    classifier_path: str = "data/char_classifier.pth"
    lambda_char: float = 0.5

    # Perceptual loss
    use_perceptual: bool = False
    lambda_perc: float = 0.1

    # Output
    output_dir: str = "outputs"
    log_every: int = 100


# ---------------------------------------------------------------------------
# Letter-pair helpers
# ---------------------------------------------------------------------------

def letter_pairs(word: str) -> list[tuple[str, str]]:
    """
    "SWIMS" → [('S','S'), ('W','M'), ('I','I')]
    "NOON"  → [('N','N'), ('O','O')]

    Pair (a, b) means the glyph reads a upright and b when rotated 180°.
    For odd-length words the middle letter is its own partner.
    """
    N = len(word)
    w = word.upper()
    return [(w[i], w[N - 1 - i]) for i in range((N + 1) // 2)]


def compose(glyphs: list[torch.Tensor], N: int) -> torch.Tensor:
    """
    Compose unique glyphs into the full N-character ambigram strip.

    The identity  rotate_180([A|B|…|Z]) = [rot(Z)|…|rot(B)|rot(A)]  means
    the layout  [g₀ | g₁ | … | gₖ₋₁ | rot(gₖ₋₂) | … | rot(g₀)]  (odd N)
            or  [g₀ | g₁ | … | gₖ₋₁ | rot(gₖ₋₁) | … | rot(g₀)]  (even N)
    is self-symmetric under 180° rotation.
    """
    k = len(glyphs)
    tiles: list[torch.Tensor] = list(glyphs)
    start = k - 2 if N % 2 == 1 else k - 1
    for i in range(start, -1, -1):
        tiles.append(rotate_180(glyphs[i]))
    return torch.cat(tiles, dim=-1)   # concat along width (last dim of CHW)


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
# Text prompts
# ---------------------------------------------------------------------------

_PROMPTS = [
    "the single letter {ch} in bold black sans-serif on a clean white background, isolated, no other marks",
    "one uppercase {ch} character, black ink on pure white, no texture or background noise",
    "capital letter {ch} only, white background, high contrast, crisp edges, completely isolated",
]


# ---------------------------------------------------------------------------
# Parallel glyph-set optimiser
# ---------------------------------------------------------------------------

class GlyphSetOptimizer:
    """
    Optimise all letter-pair glyphs simultaneously in one CLIP batch per step.

    Supports two representations:
      pixel  — direct (1, H, W) pixel tensor optimised with projected GD
      bezier — cubic Bézier stroke parameters rasterised differentiably
    """

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        config: Config,
        clip_loss: CLIPLoss,
        char_classifier: Optional[nn.Module],
        perceptual_loss: Optional[nn.Module],
        device: torch.device,
    ) -> None:
        self.pairs = pairs
        self.cfg = config
        self.clip_loss = clip_loss
        self.char_classifier = char_classifier
        self.perceptual_loss = perceptual_loss
        self.device = device

        s = config.glyph_size

        if config.mode == "pixel":
            imgs = []
            refs = []
            for char_a, char_b in pairs:
                ia = render_text_image(char_a, (s, s)).mean(0, keepdim=True).to(device)
                ib = render_text_image(char_b, (s, s)).mean(0, keepdim=True).to(device)
                ref = (ia + rotate_180(ib)) * 0.5
                refs.append(ref.detach())
                imgs.append(nn.Parameter(ref.clone()))
            self.pixel_params = nn.ParameterList(imgs)
            # Blended reference held fixed; used by the anchor loss
            self._refs: list[torch.Tensor] = refs
            self.bezier_glyphs: Optional[nn.ModuleList] = None
        else:
            self.pixel_params = None   # type: ignore[assignment]
            self._refs = []
            glyphs = []
            for char_a, char_b in pairs:
                glyphs.append(
                    BezierGlyph.from_text(
                        char_a,
                        char_b=char_b,
                        n_strokes=config.n_strokes,
                        size=s,
                        stroke_width=config.stroke_width,
                        device=device,
                        symmetric=(char_a == char_b),
                    )
                )
            self.bezier_glyphs = nn.ModuleList(glyphs)

        # Pre-compute text features once
        self.feats_a = [
            clip_loss.encode_prompts([t.format(ch=a) for t in _PROMPTS])
            for a, _ in pairs
        ]
        self.feats_b = [
            clip_loss.encode_prompts([t.format(ch=b) for t in _PROMPTS])
            for _, b in pairs
        ]

        # Perceptual target images (rendered reference letters)
        if perceptual_loss is not None:
            self.perc_targets_a = [
                render_text_image(a, (s, s)).to(device) for a, _ in pairs
            ]
            self.perc_targets_b = [
                render_text_image(b, (s, s)).to(device) for _, b in pairs
            ]
        else:
            self.perc_targets_a = []
            self.perc_targets_b = []

    def _render_all(self) -> list[torch.Tensor]:
        """Return current (1, H, W) grayscale images for all glyphs."""
        if self.pixel_params is not None:
            return list(self.pixel_params)
        assert self.bezier_glyphs is not None
        return [g.render() for g in self.bezier_glyphs]

    def run(self, save_dir: Optional[Path] = None) -> list[torch.Tensor]:
        if self.pixel_params is not None:
            params = list(self.pixel_params)
        else:
            assert self.bezier_glyphs is not None
            params = list(self.bezier_glyphs.parameters())

        opt = torch.optim.Adam(params, lr=self.cfg.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.cfg.num_steps, eta_min=self.cfg.lr * 0.1
        )

        for step in tqdm(range(self.cfg.num_steps), desc="Optimising glyphs"):
            imgs = self._render_all()                                      # list (1, H, W)
            rgbs     = [img.expand(3, -1, -1) for img in imgs]            # list (3, H, W)
            rgbs_rot = [rotate_180(img).expand(3, -1, -1) for img in imgs]

            # All glyphs × both orientations in one CLIP batch
            losses_a, losses_b = self.clip_loss.forward_all(
                rgbs, rgbs_rot, self.feats_a, self.feats_b
            )
            loss = sum(losses_a) + sum(losses_b)

            # Pixel regularisation
            if self.cfg.mode == "pixel":
                for i, img in enumerate(imgs):
                    loss = loss + self.cfg.lambda_tv * _total_variation(img)
                    loss = loss + self.cfg.lambda_bw * (img * (1.0 - img)).mean()
                    # Anchor loss: pulls toward the blended reference initialisation.
                    # This suppresses background noise/stripe artefacts that CLIP
                    # adds over time to maximise its own score.
                    loss = loss + self.cfg.lambda_anchor * (img - self._refs[i]).pow(2).mean()

            # Bézier spread: penalise stroke centres that are too close together.
            # Encourages strokes to cover the canvas instead of all piling on one spot.
            if self.cfg.mode == "bezier" and self.bezier_glyphs is not None:
                for g in self.bezier_glyphs:   # type: ignore[union-attr]
                    centres = g.control_points.mean(1)               # (n_s, 2)
                    diff = centres.unsqueeze(0) - centres.unsqueeze(1)   # (n_s, n_s, 2)
                    dists = diff.norm(dim=-1)                         # (n_s, n_s)
                    loss = loss + 0.5 * F.relu(0.12 - dists).mean()

            # Character classifier
            if self.char_classifier is not None:
                for i, (char_a, char_b) in enumerate(self.pairs):
                    loss = loss + self.cfg.lambda_char * (
                        self.char_classifier.readability_loss(imgs[i], char_a) +
                        self.char_classifier.readability_loss(rotate_180(imgs[i]), char_b)
                    )

            # Perceptual loss
            if self.perceptual_loss is not None:
                for i, img in enumerate(imgs):
                    rgb     = img.expand(3, -1, -1)
                    rgb_rot = rotate_180(img).expand(3, -1, -1)
                    loss = loss + self.cfg.lambda_perc * (
                        self.perceptual_loss(rgb,     self.perc_targets_a[i]) +
                        self.perceptual_loss(rgb_rot, self.perc_targets_b[i])
                    )

            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

            with torch.no_grad():
                if self.pixel_params is not None:
                    for idx, p in enumerate(self.pixel_params):
                        p.data.clamp_(0.0, 1.0)
                        # Hard-project background pixels back to near-white.
                        # The reference blend tells us which pixels are background
                        # (value close to 1.0). CLIP otherwise fills these with
                        # adversarial texture; clamping them post-step is cheaper
                        # and more reliable than fighting it with a loss term.
                        bg = self._refs[idx] > 0.85      # (1, H, W) bool
                        p.data[bg] = p.data[bg].clamp(0.85, 1.0)
                elif self.bezier_glyphs is not None:
                    for g in self.bezier_glyphs:
                        g.control_points.data.clamp_(0.0, 1.0)

            if save_dir and (step % self.cfg.log_every == 0 or step == self.cfg.num_steps - 1):
                with torch.no_grad():
                    for i, (char_a, char_b) in enumerate(self.pairs):
                        gd = save_dir / f"glyph_{i}_{char_a}_{char_b}"
                        gd.mkdir(exist_ok=True)
                        rgb = imgs[i].detach().cpu().expand(3, -1, -1)
                        save_comparison(rgb, gd / f"step_{step:04d}.png",
                                        word_a=char_a, word_b=char_b)

        # Final images (CPU)
        result: list[torch.Tensor] = []
        with torch.no_grad():
            for img in self._render_all():
                result.append(img.detach().cpu().expand(3, -1, -1).clone())

        # SVG export
        if save_dir and self.bezier_glyphs is not None:
            for i, (char_a, char_b) in enumerate(self.pairs):
                self.bezier_glyphs[i].to_svg(        # type: ignore[index]
                    save_dir / f"glyph_{i}_{char_a}_{char_b}.svg"
                )

        return result


# ---------------------------------------------------------------------------
# Top-level generator
# ---------------------------------------------------------------------------

class SelfAmbigramGenerator:
    """
    Generate a self-ambigram: optimise all letter-pair glyphs in parallel,
    then compose them into the final word image.
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
            use_compile=config.torch_compile,
        )

        self.pairs = letter_pairs(self.word)
        N = len(self.word)
        pair_str = ", ".join(
            f"{a}↔{b}" + (" [mid]" if N % 2 == 1 and i == len(self.pairs) - 1 else "")
            for i, (a, b) in enumerate(self.pairs)
        )
        log.info("Word: %s   Mode: %s   Pairs: %s", self.word, config.mode, pair_str)

        # Optional: character classifier
        self.char_classifier: Optional[nn.Module] = None
        if config.use_classifier:
            p = Path(config.classifier_path)
            if p.exists():
                from .char_classifier import CharClassifier
                self.char_classifier = CharClassifier.load(p, device)
                log.info("Loaded character classifier from %s", p)
            else:
                log.warning("Classifier checkpoint not found at %s — skipping", p)

        # Optional: perceptual loss
        self.perceptual_loss: Optional[nn.Module] = None
        if config.use_perceptual:
            from .losses.perceptual_loss import PerceptualLoss
            self.perceptual_loss = PerceptualLoss(device)
            log.info("Perceptual (VGG) loss enabled.")

    def run(self) -> torch.Tensor:
        out = Path(self.cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        optimizer = GlyphSetOptimizer(
            pairs=self.pairs,
            config=self.cfg,
            clip_loss=self.clip_loss,
            char_classifier=self.char_classifier,
            perceptual_loss=self.perceptual_loss,
            device=self.device,
        )
        glyphs = optimizer.run(save_dir=out)

        # Save individual glyph images
        for i, ((char_a, char_b), glyph) in enumerate(zip(self.pairs, glyphs)):
            save_image(glyph, out / f"glyph_{i}_{char_a}_{char_b}.png")
            save_image(rotate_180(glyph), out / f"glyph_{i}_{char_a}_{char_b}_rot.png")

        composed = compose(glyphs, len(self.word))
        save_comparison(composed, out / "final.png", word_a=self.word, word_b=self.word)
        save_image(composed, out / "final_upright.png")
        save_image(rotate_180(composed), out / "final_rotated.png")

        log.info("Done → %s/final.png", out)
        return composed
