from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from .losses.sds import SDSLoss
from .utils.image import (
    blend_init,
    logit,
    rotate_180,
    save_comparison,
    save_image,
)

log = logging.getLogger(__name__)


@dataclass
class OptimizerConfig:
    # Words
    word_a: str = "love"
    word_b: str = ""          # empty → self-ambigram (same as word_a)

    # Image
    image_size: int = 512
    image_channels: int = 3

    # Prompt templates
    prompt_template: str = "the word '{word}' in clear bold typography on white background"
    neg_prompt: str = "blurry, noisy, illegible, low quality, watermark"

    # Optimization
    num_steps: int = 1500
    lr: float = 5e-3
    weight_decay: float = 0.0
    lambda_b: float = 1.0     # relative weight of the rotated-view SDS loss

    # SDS hyperparameters
    guidance_scale: float = 7.5
    min_t: int = 50
    max_t: int = 950

    # Timestep annealing: linearly shrink [min_t, max_t] → [min_t, anneal_max_t]
    # Set anneal_max_t = max_t to disable.
    anneal_max_t: int = 400
    anneal_start: float = 0.5   # fraction of steps before annealing begins

    # Logging
    output_dir: str = "outputs"
    log_every: int = 100
    use_wandb: bool = False


class AmbigramOptimizer:
    """
    Generates an ambigram image by jointly minimizing two SDS losses:
      - Loss A: image (upright)     should look like word_a
      - Loss B: image (rotated 180°) should look like word_b
    """

    def __init__(
        self,
        config: OptimizerConfig,
        pipeline,          # diffusers StableDiffusionPipeline (already on device)
        device: torch.device,
    ) -> None:
        self.cfg = config
        self.pipe = pipeline
        self.device = device

        word_b = config.word_b or config.word_a
        self.word_a = config.word_a
        self.word_b = word_b

        self.sds = SDSLoss(
            unet=pipeline.unet,
            scheduler=pipeline.scheduler,
            min_t=config.min_t,
            max_t=config.max_t,
            guidance_scale=config.guidance_scale,
        ).to(device)

        # Pre-compute frozen text embeddings
        self.emb_a, self.emb_uncond = self._encode_prompts(word_a=config.word_a)
        self.emb_b, _ = self._encode_prompts(word_a=word_b)

        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> torch.Tensor:
        """Run the full optimization and return the final image (CHW, [0,1])."""
        raw = self._init_raw_image()
        opt = torch.optim.Adam([raw], lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.num_steps)

        if self.cfg.use_wandb:
            import wandb
            wandb.init(
                project="ambigrams",
                config=vars(self.cfg),
                name=f"{self.word_a}__{self.word_b}",
            )

        pbar = tqdm(range(self.cfg.num_steps), desc=f"'{self.word_a}' / '{self.word_b}'")
        for step in pbar:
            self._update_sds_range(step)

            image = torch.sigmoid(raw)                     # [0, 1]
            image_norm = image * 2.0 - 1.0                 # [-1, 1] for VAE

            # Upright view → word_a
            latent_a = self._encode_image(image_norm)
            loss_a = self.sds(latent_a, self.emb_a, self.emb_uncond)

            # Rotated view → word_b
            image_rot = rotate_180(image_norm)
            latent_b = self._encode_image(image_rot)
            loss_b = self.sds(latent_b, self.emb_b, self.emb_uncond)

            loss = loss_a + self.cfg.lambda_b * loss_b

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([raw], max_norm=1.0)
            opt.step()
            sched.step()

            pbar.set_postfix(loss_a=f"{loss_a.item():.3f}", loss_b=f"{loss_b.item():.3f}")

            if self.cfg.use_wandb:
                import wandb
                wandb.log({"loss_a": loss_a.item(), "loss_b": loss_b.item(), "step": step})

            if step % self.cfg.log_every == 0 or step == self.cfg.num_steps - 1:
                save_comparison(
                    image,
                    f"{self.cfg.output_dir}/step_{step:05d}.png",
                    word_a=self.word_a,
                    word_b=self.word_b,
                )

        final = torch.sigmoid(raw).detach().cpu().squeeze(0)
        save_comparison(
            final,
            f"{self.cfg.output_dir}/final.png",
            word_a=self.word_a,
            word_b=self.word_b,
        )
        save_image(final, f"{self.cfg.output_dir}/final_upright.png")
        save_image(rotate_180(final), f"{self.cfg.output_dir}/final_rotated.png")

        if self.cfg.use_wandb:
            import wandb
            wandb.finish()

        log.info("Saved to %s/", self.cfg.output_dir)
        return final

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_raw_image(self) -> nn.Parameter:
        """
        Initialise from a blended text render (word_a upright + word_b rotated).
        This gives a better starting point than random noise.
        """
        size = (self.cfg.image_size, self.cfg.image_size)
        blended = blend_init(self.word_a, self.word_b, size=size)    # CHW [0,1]
        blended = blended.unsqueeze(0).to(self.device)               # 1CHW
        raw = logit(blended)
        return nn.Parameter(raw)

    @torch.no_grad()
    def _encode_prompts(self, word_a: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cond_emb, uncond_emb) for the word prompt."""
        prompt = self.cfg.prompt_template.format(word=word_a)

        tok_cond = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(self.device)

        tok_uncond = self.pipe.tokenizer(
            self.cfg.neg_prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(self.device)

        emb_cond = self.pipe.text_encoder(tok_cond)[0]
        emb_uncond = self.pipe.text_encoder(tok_uncond)[0]
        return emb_cond, emb_uncond

    def _encode_image(self, image_norm: torch.Tensor) -> torch.Tensor:
        """
        Encode a [-1, 1] pixel image to VAE latent space (with gradient).
        The VAE may be fp16; we cast in/out so the optimizer stays in fp32.
        """
        vae_dtype = next(self.pipe.vae.parameters()).dtype
        latent_dist = self.pipe.vae.encode(image_norm.to(dtype=vae_dtype))
        latent = latent_dist.latent_dist.sample()
        return (latent * self.pipe.vae.config.scaling_factor).float()

    def _update_sds_range(self, step: int) -> None:
        """Anneal the maximum timestep as optimization progresses."""
        if self.cfg.anneal_max_t >= self.cfg.max_t:
            return
        frac = step / self.cfg.num_steps
        if frac < self.cfg.anneal_start:
            return
        progress = (frac - self.cfg.anneal_start) / (1.0 - self.cfg.anneal_start)
        new_max = int(self.cfg.max_t + progress * (self.cfg.anneal_max_t - self.cfg.max_t))
        self.sds.max_t = max(self.cfg.min_t + 10, new_max)
