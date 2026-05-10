from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .losses.clip_loss import CLIPLoss
from .utils.image import blend_init, rotate_180, save_comparison, save_image

log = logging.getLogger(__name__)


@dataclass
class CLIPOptimizerConfig:
    # Words
    word_a: str = "love"
    word_b: str = ""          # empty → self-ambigram

    # Image
    image_size: int = 512

    # CLIP model
    clip_model: str = "ViT-L-14"
    clip_pretrained: str = "openai"
    n_augments: int = 16      # crops per orientation; more = stabler gradients but slower

    # Optimisation
    num_steps: int = 2000
    lr: float = 2e-2

    # Loss weights
    lambda_b: float = 1.0    # weight of the rotated-view CLIP loss
    lambda_tv: float = 5e-4  # total variation — controls smoothness
    lambda_bw: float = 0.1   # black/white push — encourages crisp typography

    # Multi-prompt templates — averaged for robustness
    prompt_templates: list[str] = field(default_factory=lambda: [
        "the word '{word}' in black text on a white background",
        "text that reads '{word}'",
        "the letters {letters} written out",
    ])

    # Output
    output_dir: str = "outputs"
    log_every: int = 100


def _total_variation(x: torch.Tensor) -> torch.Tensor:
    """Anisotropic TV over a (C, H, W) or (1, C, H, W) tensor."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw


class CLIPAmbigramOptimizer:
    """
    Generates a 180° rotational ambigram by optimising a pixel image to simultaneously
    maximise CLIP similarity for word_a (upright) and word_b (rotated 180°).

    Uses projected gradient descent: image lives in [0, 1] and is clamped back after
    each Adam step, avoiding the sigmoid saturation that killed the SDS approach.
    """

    def __init__(
        self,
        config: CLIPOptimizerConfig,
        device: torch.device,
    ) -> None:
        self.cfg = config
        self.device = device

        self.word_a = config.word_a
        self.word_b = config.word_b or config.word_a

        log.info("Loading CLIP %s (%s)…", config.clip_model, config.clip_pretrained)
        self.loss_fn = CLIPLoss(
            model_name=config.clip_model,
            pretrained=config.clip_pretrained,
            device=device,
            n_augments=config.n_augments,
        )

        self.feat_a = self._encode(self.word_a)
        self.feat_b = self._encode(self.word_b)

        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> torch.Tensor:
        image = self._init_image()          # leaf tensor in [0,1] with grad
        optimizer = torch.optim.Adam([image], lr=self.cfg.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.num_steps, eta_min=self.cfg.lr * 0.1
        )

        pbar = tqdm(range(self.cfg.num_steps), desc=f"'{self.word_a}' / '{self.word_b}'")
        for step in pbar:
            # Both orientations in one CLIP batch — half the forward passes
            loss_a, loss_b = self.loss_fn.forward_pair(
                image, rotate_180(image), self.feat_a, self.feat_b
            )
            loss_tv = _total_variation(image)
            loss_bw = (image * (1.0 - image)).mean()

            loss = (loss_a
                    + self.cfg.lambda_b  * loss_b
                    + self.cfg.lambda_tv * loss_tv
                    + self.cfg.lambda_bw * loss_bw)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                image.data.clamp_(0.0, 1.0)

            pbar.set_postfix(
                a=f"{loss_a.item():.3f}",
                b=f"{loss_b.item():.3f}",
                tv=f"{loss_tv.item():.4f}",
            )

            if step % self.cfg.log_every == 0 or step == self.cfg.num_steps - 1:
                self._save(image.detach().cpu(), f"step_{step:05d}.png")

        final = image.detach().cpu()
        self._save(final, "final.png")
        save_image(final, f"{self.cfg.output_dir}/final_upright.png")
        save_image(rotate_180(final), f"{self.cfg.output_dir}/final_rotated.png")
        log.info("Done. Saved to %s/", self.cfg.output_dir)
        return final

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_image(self) -> torch.Tensor:
        size = (self.cfg.image_size, self.cfg.image_size)
        image = blend_init(self.word_a, self.word_b, size=size)
        image = image.to(self.device).detach().clone()
        image.requires_grad_(True)
        return image

    def _encode(self, word: str) -> torch.Tensor:
        prompts = [
            t.format(word=word, letters=" ".join(word.upper()))
            for t in self.cfg.prompt_templates
        ]
        return self.loss_fn.encode_prompts(prompts)     # (1, d)

    def _save(self, image: torch.Tensor, filename: str) -> None:
        save_comparison(
            image,
            f"{self.cfg.output_dir}/{filename}",
            word_a=self.word_a,
            word_b=self.word_b,
        )
