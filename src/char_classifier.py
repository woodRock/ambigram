from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharClassifier(nn.Module):
    """
    Small CNN: (B, 1, H, W) grayscale glyph → (B, 26) logits for A-Z.
    Designed for use as a differentiable readability signal during optimization.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 26),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    @staticmethod
    def char_index(ch: str) -> int:
        return ord(ch.upper()) - ord("A")

    def readability_loss(self, image: torch.Tensor, char: str) -> torch.Tensor:
        """Cross-entropy loss driving image to look like the given character."""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        logits = self.forward(image)
        target = torch.tensor([self.char_index(char)], device=image.device)
        return F.cross_entropy(logits, target)

    @classmethod
    def load(cls, path: str | Path, device: torch.device) -> "CharClassifier":
        model = cls().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model
