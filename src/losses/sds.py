from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SDSLoss(nn.Module):
    """
    Score Distillation Sampling loss (DreamFusion, Poole et al. 2022).

    Given a latent z that is differentiably produced from the optimizable image,
    this loss returns a scalar whose gradient pushes z toward the distribution
    described by `text_emb`, using the frozen diffusion UNet as the score estimator.
    """

    def __init__(
        self,
        unet: nn.Module,
        scheduler,
        min_t: int = 50,
        max_t: int = 950,
        guidance_scale: float = 7.5,
    ) -> None:
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler
        self.min_t = min_t
        self.max_t = max_t
        self.guidance_scale = guidance_scale

        # Cache alphas on the same device as the scheduler
        alphas: torch.Tensor = scheduler.alphas_cumprod.float()
        self.register_buffer("alphas_cumprod", alphas)

    def forward(
        self,
        latent: torch.Tensor,
        text_emb: torch.Tensor,
        uncond_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            latent:    (1, 4, h, w)  — must have grad enabled
            text_emb:  (1, seq, d)   — frozen text embedding for the target prompt
            uncond_emb:(1, seq, d)   — empty-string embedding for CFG
        Returns:
            scalar loss whose .backward() provides the SDS gradient
        """
        device = latent.device
        bsz = latent.shape[0]

        t = torch.randint(self.min_t, self.max_t, (bsz,), device=device, dtype=torch.long)

        noise = torch.randn_like(latent)
        noisy = self.scheduler.add_noise(latent, noise, t)

        # Cast to UNet dtype (may be fp16) for the frozen forward pass
        unet_dtype = next(self.unet.parameters()).dtype
        latent_in = torch.cat([noisy, noisy]).to(dtype=unet_dtype)
        emb_in = torch.cat([uncond_emb.expand(bsz, -1, -1),
                             text_emb.expand(bsz, -1, -1)]).to(dtype=unet_dtype)

        with torch.no_grad():
            noise_pred = self.unet(latent_in, torch.cat([t, t]), encoder_hidden_states=emb_in).sample

        # Back to fp32 for gradient arithmetic
        noise_u, noise_c = noise_pred.float().chunk(2)
        noise_guided = noise_u + self.guidance_scale * (noise_c - noise_u)

        # Weighting: w(t) = 1 - ᾱ_t  (variance of the noising process)
        alpha_t = self.alphas_cumprod[t].to(device).view(-1, 1, 1, 1)
        w = (1.0 - alpha_t)

        # Phantom-gradient target: treat (ε̂ - ε) as a constant w.r.t. latent
        grad = w * (noise_guided - noise)
        grad = grad.clamp(-1.0, 1.0)  # stability

        # MSE against stop-gradient target — gradient flows through `latent` only
        target = (latent - grad).detach()
        loss = 0.5 * F.mse_loss(latent, target, reduction="sum")
        return loss
