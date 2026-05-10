from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from .utils.image import blend_init_pil, rotate_180_pil, save_comparison_pil

log = logging.getLogger(__name__)


@dataclass
class Img2ImgConfig:
    word_a: str = "love"
    word_b: str = ""          # empty → self-ambigram

    image_size: int = 512

    prompt_template: str = (
        "the word '{word}' in bold black sans-serif typography on a plain white background, "
        "high contrast, sharp edges, perfectly legible"
    )
    neg_prompt: str = (
        "blurry, noisy, illegible, low quality, watermark, artistic, decorative, "
        "handwriting, multiple words, letters overlapping"
    )

    # Alternating refinement schedule
    num_rounds: int = 40          # total alternating passes (each round = upright + rotated)
    num_inference_steps: int = 40
    guidance_scale: float = 10.0

    # Strength annealing: start broad (allow big changes), finish fine (preserve structure)
    initial_strength: float = 0.65
    final_strength: float = 0.25

    # How much to mix the img2img result back with the pre-pass image.
    # 1.0 = fully accept result, 0.5 = half-half. Keeps each view from erasing the other.
    blend_alpha: float = 0.85

    output_dir: str = "outputs"
    save_every: int = 5


class AlternatingImg2ImgGenerator:
    """
    Generates a rotational ambigram via alternating img2img refinement.

    Each round:
      1. Run img2img on the current image with a prompt for word_a (upright).
      2. Rotate the result 180°.
      3. Run img2img with a prompt for word_b.
      4. Rotate back 180°.

    Strength is annealed from initial_strength → final_strength so early rounds
    establish coarse structure and later rounds sharpen detail without destroying
    the ambigram constraint.
    """

    def __init__(self, config: Img2ImgConfig, pipeline, device) -> None:
        self.cfg = config
        self.device = device

        word_b = config.word_b or config.word_a
        self.word_a = config.word_a
        self.word_b = word_b

        # Re-use the components from the text-to-image pipeline
        from diffusers import StableDiffusionImg2ImgPipeline
        self.pipe = StableDiffusionImg2ImgPipeline(
            vae=pipeline.vae,
            text_encoder=pipeline.text_encoder,
            tokenizer=pipeline.tokenizer,
            unet=pipeline.unet,
            scheduler=pipeline.scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        ).to(device)
        self.pipe.set_progress_bar_config(disable=True)

        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def run(self) -> Image.Image:
        size = (self.cfg.image_size, self.cfg.image_size)
        image = blend_init_pil(self.word_a, self.word_b, size=size)

        save_comparison_pil(
            image,
            f"{self.cfg.output_dir}/round_000.png",
            word_a=self.word_a,
            word_b=self.word_b,
        )

        for round_idx in tqdm(range(self.cfg.num_rounds), desc=f"'{self.word_a}'/'{self.word_b}'"):
            strength = self._strength(round_idx)

            # --- pass 1: push upright view toward word_a ---
            refined_a = self._denoise(image, self.word_a, strength)
            image = Image.blend(image, refined_a, self.cfg.blend_alpha)

            # --- pass 2: rotate, push toward word_b, rotate back ---
            image_rot = rotate_180_pil(image)
            refined_b = self._denoise(image_rot, self.word_b, strength)
            image_rot = Image.blend(image_rot, refined_b, self.cfg.blend_alpha)
            image = rotate_180_pil(image_rot)

            if (round_idx + 1) % self.cfg.save_every == 0:
                save_comparison_pil(
                    image,
                    f"{self.cfg.output_dir}/round_{round_idx + 1:03d}.png",
                    word_a=self.word_a,
                    word_b=self.word_b,
                )

        # Final outputs
        save_comparison_pil(image, f"{self.cfg.output_dir}/final.png",
                            word_a=self.word_a, word_b=self.word_b)
        image.save(f"{self.cfg.output_dir}/final_upright.png")
        rotate_180_pil(image).save(f"{self.cfg.output_dir}/final_rotated.png")

        log.info("Saved to %s/", self.cfg.output_dir)
        return image

    # ------------------------------------------------------------------

    def _strength(self, round_idx: int) -> float:
        """Linear anneal from initial_strength → final_strength."""
        t = round_idx / max(self.cfg.num_rounds - 1, 1)
        return self.cfg.initial_strength + t * (self.cfg.final_strength - self.cfg.initial_strength)

    def _denoise(self, image: Image.Image, word: str, strength: float) -> Image.Image:
        prompt = self.cfg.prompt_template.format(word=word)
        result = self.pipe(
            prompt=prompt,
            image=image,
            strength=strength,
            guidance_scale=self.cfg.guidance_scale,
            num_inference_steps=self.cfg.num_inference_steps,
            negative_prompt=self.cfg.neg_prompt,
        ).images[0]
        return result
