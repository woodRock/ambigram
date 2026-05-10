from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """VGG16 perceptual loss: MSE on relu1_2 and relu2_2 feature maps."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        # relu1_2 = layers 0-3, relu2_2 = layers 0-8
        self.slice1 = nn.Sequential(*list(vgg.children())[:4]).to(device).eval()
        self.slice2 = nn.Sequential(*list(vgg.children())[:9]).to(device).eval()
        for p in self.parameters():
            p.requires_grad_(False)
        self.device = device

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """MSE over two VGG feature levels. Both inputs: (C, H, W) or (B, C, H, W) in [0,1]."""
        if x.dim() == 3:
            x = x.unsqueeze(0)
            target = target.unsqueeze(0)
        return (
            (self.slice1(x) - self.slice1(target)).pow(2).mean() +
            (self.slice2(x) - self.slice2(target)).pow(2).mean()
        )
