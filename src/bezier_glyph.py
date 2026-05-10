from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Differentiable Bézier rasteriser
# ---------------------------------------------------------------------------

def _bezier3_samples(curves: torch.Tensor, n_samples: int = 32) -> torch.Tensor:
    """
    Sample cubic Bézier curves at n_samples uniform t values.

    curves: (n_strokes, 4, 2)  — control points in [0, 1] canvas coords (x, y)
    returns: (n_strokes, n_samples, 2)
    """
    ts = torch.linspace(0.0, 1.0, n_samples, device=curves.device)   # (T,)
    t = ts[None, :, None]     # (1, T, 1)
    mt = 1.0 - t
    p = [curves[:, k:k+1, :] for k in range(4)]                      # each (n, 1, 2)
    return mt**3*p[0] + 3*mt**2*t*p[1] + 3*mt*t**2*p[2] + t**3*p[3] # (n, T, 2)


def _stroke_coverage(
    px: torch.Tensor,       # (H*W, 2)  pixel grid coords in [0, 1]
    curves: torch.Tensor,   # (n_strokes, 4, 2)
    width: float,           # stroke half-width in [0, 1] canvas units
    n_samples: int = 32,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """
    Differentiable coverage map.

    For each pixel finds the soft-min distance to all curve samples, then
    applies a sigmoid to get per-pixel coverage ∈ [0, 1].
    Chunked to keep peak memory under ~50 MB even for 256×256 inputs.
    Returns (H*W,) in [0, 1].
    """
    pts_flat = _bezier3_samples(curves, n_samples).reshape(-1, 2)   # (n*T, 2)
    HW = px.shape[0]
    sharpness = float(n_samples) * 8.0   # soft-min picks nearest sample

    parts: list[torch.Tensor] = []
    for start in range(0, HW, chunk_size):
        chunk = px[start : start + chunk_size]                         # (c, 2)
        diff = chunk.unsqueeze(1) - pts_flat.unsqueeze(0)              # (c, n*T, 2)
        dist = diff.norm(dim=-1)                                       # (c, n*T)
        # logsumexp-based soft-min
        min_d = -torch.logsumexp(-sharpness * dist, dim=-1) / sharpness  # (c,)
        parts.append(min_d)

    min_dist = torch.cat(parts)                                        # (H*W,)
    # Sigmoid transition centred at stroke edge; steep but smooth for gradient flow
    return torch.sigmoid((width - min_dist) * (8.0 / width))


# ---------------------------------------------------------------------------
# BezierGlyph
# ---------------------------------------------------------------------------

class BezierGlyph(nn.Module):
    """
    A glyph parameterised as a set of cubic Bézier strokes.

    Learnable parameters: control_points (n_strokes, 4, 2) ∈ [0, 1].
    render() → (1, H, W) float32 in [0, 1] — white background, black strokes.
    """

    def __init__(
        self,
        n_strokes: int = 10,
        size: int = 256,
        stroke_width: float = 0.04,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.size = size
        self.stroke_width = stroke_width

        # Random init in the central 60 % of the canvas
        pts = torch.rand(n_strokes, 4, 2, device=device) * 0.6 + 0.2
        self.control_points = nn.Parameter(pts)

        # Pre-compute the pixel coordinate grid — stays fixed, no grad needed
        ys = torch.linspace(0.0, 1.0, size, device=device)
        xs = torch.linspace(0.0, 1.0, size, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer(
            "pixel_grid",
            torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2),
        )

    @classmethod
    def from_text(
        cls,
        char: str,
        n_strokes: int = 10,
        size: int = 256,
        stroke_width: float = 0.04,
        device: torch.device = torch.device("cpu"),
    ) -> "BezierGlyph":
        """
        Initialise with curves whose endpoints are sampled from the rendered letter's
        ink pixels, giving the optimiser a head start versus pure random init.
        """
        from .utils.image import render_text_image

        img = render_text_image(char, (size, size)).mean(0)    # (H, W) grayscale
        ink = (img < 0.4).nonzero(as_tuple=False).float()     # (N, 2) as (row, col)

        glyph = cls(n_strokes, size, stroke_width, device)

        if ink.shape[0] < 4:
            return glyph

        ink_xy = ink[:, [1, 0]] / (size - 1)   # (N, 2) as (x, y) in [0, 1]

        pts_list = []
        for _ in range(n_strokes):
            idxs = torch.randperm(ink_xy.shape[0])[:2]
            p0, p3 = ink_xy[idxs[0]], ink_xy[idxs[1]]
            noise = torch.randn(2, 2) * 0.04
            p1 = (p0 * 0.67 + p3 * 0.33 + noise[0]).clamp(0.0, 1.0)
            p2 = (p0 * 0.33 + p3 * 0.67 + noise[1]).clamp(0.0, 1.0)
            pts_list.append(torch.stack([p0, p1, p2, p3]))

        glyph.control_points = nn.Parameter(
            torch.stack(pts_list).clamp(0.0, 1.0).to(device)
        )
        return glyph

    def render(self) -> torch.Tensor:
        """Returns (1, H, W) in [0, 1] — white background, black strokes."""
        pts = self.control_points.clamp(0.0, 1.0)
        coverage = _stroke_coverage(
            self.pixel_grid, pts, width=self.stroke_width  # type: ignore[arg-type]
        )
        img = 1.0 - coverage.reshape(self.size, self.size)
        return img.unsqueeze(0)   # (1, H, W)

    def to_svg(self, path: str | Path) -> None:
        """Export the Bézier strokes as an SVG file."""
        s = self.size
        sw = self.stroke_width * s
        pts = self.control_points.detach().cpu().numpy() * s

        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{s}" height="{s}" viewBox="0 0 {s} {s}">',
            f'<rect width="{s}" height="{s}" fill="white"/>',
        ]
        for curve in pts:
            p0, p1, p2, p3 = curve
            d = (
                f"M {p0[0]:.2f},{p0[1]:.2f} "
                f"C {p1[0]:.2f},{p1[1]:.2f} {p2[0]:.2f},{p2[1]:.2f} "
                f"{p3[0]:.2f},{p3[1]:.2f}"
            )
            lines.append(
                f'<path d="{d}" fill="none" stroke="black" '
                f'stroke-width="{sw:.2f}" stroke-linecap="round" stroke-linejoin="round"/>'
            )
        lines.append("</svg>")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("\n".join(lines))
