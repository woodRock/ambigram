from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import open_clip

_MEAN = (0.48145466, 0.4578275,  0.40821073)
_STD  = (0.26862954, 0.26130258, 0.27577711)


class CLIPLoss(nn.Module):
    """
    Differentiable CLIP readability loss.

    Applies n_augments random crops per image, encodes them in one batch, and
    returns mean negative cosine similarity against pre-computed text features.

    Large minimum crop scale (0.65) prevents the optimiser from satisfying CLIP
    with background texture fragments instead of improving the letterforms.
    No flips are used — they would mirror text characters.
    """

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: torch.device = torch.device("cpu"),
        n_augments: int = 16,
        use_compile: bool = False,
    ) -> None:
        super().__init__()

        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        model = model.to(device).eval()
        model.requires_grad_(False)

        if use_compile and hasattr(torch, "compile") and device.type == "cuda":
            model = torch.compile(model)

        self.clip = model
        self.n_augments = n_augments
        self.device = device
        self.normalize = T.Normalize(mean=_MEAN, std=_STD)

        self.augment = T.Compose([
            T.RandomResizedCrop(
                224,
                scale=(0.65, 1.0),
                ratio=(0.85, 1.15),
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            T.RandomAffine(
                degrees=3,
                translate=(0.03, 0.03),
                interpolation=T.InterpolationMode.BILINEAR,
            ),
        ])

    @torch.no_grad()
    def encode_prompts(self, prompts: list[str]) -> torch.Tensor:
        """Average-pool normalised text features over a list of prompts → (1, d)."""
        tokens = open_clip.tokenize(prompts).to(self.device)
        features = self.clip.encode_text(tokens)
        features = F.normalize(features, dim=-1)
        return features.mean(0, keepdim=True)

    def _crops(self, img: torch.Tensor) -> torch.Tensor:
        """Stack n_augments augmented crops from one image: (n_aug, 3, 224, 224)."""
        if img.dim() == 4:
            img = img.squeeze(0)
        return torch.stack([self.normalize(self.augment(img)) for _ in range(self.n_augments)])

    def forward_pair(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate losses for both orientations in one CLIP forward pass.
        Returns (loss_a, loss_b) — negative cosine similarities.
        """
        batch = torch.cat([self._crops(image_a), self._crops(image_b)])  # (2*n, 3, 224, 224)

        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            img_features = self.clip.encode_image(batch)

        img_features = F.normalize(img_features.float(), dim=-1)
        n = self.n_augments
        fa, fb = img_features[:n], img_features[n:]
        return -(fa @ feat_a.T).mean(), -(fb @ feat_b.T).mean()

    def forward_all(
        self,
        images: list[torch.Tensor],
        images_rot: list[torch.Tensor],
        feats_a: list[torch.Tensor],
        feats_b: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Process all glyphs (both orientations) in a single CLIP forward pass.

        Concatenates n_augments crops for each upright image followed by n_augments
        crops for each rotated image — batch size = 2 * n_glyphs * n_augments.
        This amortises CLIP's fixed overhead across all glyphs each step.

        Returns (losses_a, losses_b) — one scalar per glyph.
        """
        n_g = len(images)
        n = self.n_augments
        upright_crops = [self._crops(img) for img in images]      # n_g × (n, 3, 224, 224)
        rotated_crops = [self._crops(img) for img in images_rot]  # n_g × (n, 3, 224, 224)
        batch = torch.cat(upright_crops + rotated_crops)           # (2*n_g*n, 3, 224, 224)

        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            features = self.clip.encode_image(batch)

        features = F.normalize(features.float(), dim=-1)           # (2*n_g*n, d)

        losses_a: list[torch.Tensor] = []
        losses_b: list[torch.Tensor] = []
        for i in range(n_g):
            fa = features[i * n : (i + 1) * n]
            fb = features[(n_g + i) * n : (n_g + i + 1) * n]
            losses_a.append(-(fa @ feats_a[i].T).mean())
            losses_b.append(-(fb @ feats_b[i].T).mean())

        return losses_a, losses_b
