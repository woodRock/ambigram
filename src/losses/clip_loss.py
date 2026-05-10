from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import open_clip

# Standard CLIP normalisation constants
_MEAN = (0.48145466, 0.4578275, 0.40821073)
_STD  = (0.26862954, 0.26130258, 0.27577711)


class CLIPLoss(nn.Module):
    """
    Differentiable CLIP readability loss.

    For each forward call, applies `n_augments` random crops to the input image,
    runs all crops through the frozen CLIP encoder in one batch, then returns the
    mean negative cosine similarity with the pre-computed text features.

    Averaging over many augmentations prevents the optimiser finding adversarial
    single-crop solutions that fool CLIP but look like noise to a human.
    No horizontal/vertical flips are used — they would mirror text characters.
    """

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: torch.device = torch.device("cpu"),
        n_augments: int = 32,
    ) -> None:
        super().__init__()

        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.clip = model.to(device).eval()
        self.clip.requires_grad_(False)

        self.n_augments = n_augments
        self.device = device
        self.normalize = T.Normalize(mean=_MEAN, std=_STD)

        # Random crop + mild affine — no flips, no colour inversion
        self.augment = T.Compose([
            T.RandomResizedCrop(
                224,
                scale=(0.4, 1.0),
                ratio=(0.75, 1.33),
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            T.RandomAffine(
                degrees=5,
                translate=(0.05, 0.05),
                scale=(0.9, 1.1),
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

    def forward(self, image: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image:         (C, H, W) float32 in [0, 1] — must require grad
            text_features: (1, d)   pre-computed normalised text embedding
        Returns:
            scalar — lower means the image matches the text prompt better
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)          # (1, C, H, W)

        crops = torch.stack([
            self.normalize(self.augment(image.squeeze(0)))
            for _ in range(self.n_augments)
        ])                                       # (n_aug, 3, 224, 224)

        img_features = self.clip.encode_image(crops)
        img_features = F.normalize(img_features, dim=-1)   # (n_aug, d)

        similarity = (img_features @ text_features.T).mean()
        return -similarity
