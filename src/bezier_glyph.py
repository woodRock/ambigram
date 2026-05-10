from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# SVG tapered stroke utility (shared with svg_optimizer)
# ---------------------------------------------------------------------------

def tapered_path_svg(curve: np.ndarray, max_sw: float, n_samples: int = 24) -> str:
    """
    Filled SVG path for one cubic Bézier stroke with sin-taper width.
    Width is 0 at both ends and max_sw at the midpoint — like a brush stroke.
    curve: (4, 2) control points in pixel coordinates.
    """
    p0, p1, p2, p3 = curve
    ts = np.linspace(0.0, 1.0, n_samples)
    mt = 1.0 - ts

    bx = mt**3*p0[0] + 3*mt**2*ts*p1[0] + 3*mt*ts**2*p2[0] + ts**3*p3[0]
    by = mt**3*p0[1] + 3*mt**2*ts*p1[1] + 3*mt*ts**2*p2[1] + ts**3*p3[1]

    dx = 3*(mt**2*(p1[0]-p0[0]) + 2*mt*ts*(p2[0]-p1[0]) + ts**2*(p3[0]-p2[0]))
    dy = 3*(mt**2*(p1[1]-p0[1]) + 2*mt*ts*(p2[1]-p1[1]) + ts**2*(p3[1]-p2[1]))

    norms  = np.sqrt(dx**2 + dy**2) + 1e-8
    px, py = -dy / norms, dx / norms          # perpendicular unit vector

    w  = max_sw * np.sin(np.pi * ts) + 0.5   # taper; +0.5 keeps ends thin but nonzero

    ux, uy = bx + px * w, by + py * w
    lx, ly = bx - px * w, by - py * w

    pts = list(zip(ux, uy)) + list(zip(lx[::-1], ly[::-1]))
    d   = f"M {pts[0][0]:.1f},{pts[0][1]:.1f}" + "".join(f" L {x:.1f},{y:.1f}" for x, y in pts[1:]) + " Z"
    return f'<path d="{d}" fill="black" stroke="none"/>'


# ---------------------------------------------------------------------------
# Differentiable Bézier rasteriser
# ---------------------------------------------------------------------------

def _bezier3_samples(curves: torch.Tensor, n_samples: int = 32) -> torch.Tensor:
    """
    Sample cubic Bézier curves at n_samples uniform t values.

    curves: (n_strokes, 4, 2)  — control points in [0, 1] canvas coords (x, y)
    returns: (n_strokes, n_samples, 2)
    """
    ts = torch.linspace(0.0, 1.0, n_samples, device=curves.device)
    t  = ts[None, :, None]
    mt = 1.0 - t
    p  = [curves[:, k:k+1, :] for k in range(4)]
    return mt**3*p[0] + 3*mt**2*t*p[1] + 3*mt*t**2*p[2] + t**3*p[3]  # (n, T, 2)


def _stroke_coverage(
    px:         torch.Tensor,   # (H*W, 2)  pixel grid coords in [0, 1]
    curves:     torch.Tensor,   # (n_strokes, 4, 2)
    width:      float,
    n_samples:  int  = 32,
    chunk_size: int  = 2048,
) -> torch.Tensor:
    """
    Differentiable coverage map — returns (H*W,) in [0, 1].
    Uses a chunked soft-min over all curve samples to keep peak memory low.
    """
    pts_flat  = _bezier3_samples(curves, n_samples).reshape(-1, 2)
    HW        = px.shape[0]
    sharpness = float(n_samples) * 8.0

    parts: list[torch.Tensor] = []
    for start in range(0, HW, chunk_size):
        chunk = px[start : start + chunk_size]
        diff  = chunk.unsqueeze(1) - pts_flat.unsqueeze(0)
        dist  = diff.norm(dim=-1)
        min_d = -torch.logsumexp(-sharpness * dist, dim=-1) / sharpness
        parts.append(min_d)

    min_dist = torch.cat(parts)
    return torch.sigmoid((width - min_dist) * (8.0 / width))


# ---------------------------------------------------------------------------
# BezierGlyph
# ---------------------------------------------------------------------------

class BezierGlyph(nn.Module):
    """
    A glyph parameterised as cubic Bézier strokes.

    symmetric=True (for self-pairs like S↔S):
        Only n_strokes//2 "base" control-point sets are stored as learnable
        parameters.  The other half are their 180° mirrors, computed on the
        fly as  mirror[i] = (1 − base[i]).flip(control-point axis).
        This guarantees the rendered image is identical when rotated 180°,
        which is the exact constraint a self-pair glyph must satisfy.

    symmetric=False (for cross-pairs like J↔E):
        All n_strokes are free parameters; the optimiser finds the shape
        that reads as char_a upright and char_b rotated.
    """

    def __init__(
        self,
        n_strokes:    int          = 4,
        size:         int          = 256,
        stroke_width: float        = 0.04,
        device:       torch.device = torch.device("cpu"),
        symmetric:    bool         = False,
    ) -> None:
        super().__init__()
        self.size         = size
        self.stroke_width = stroke_width
        self.symmetric    = symmetric
        self.n_strokes    = n_strokes
        # Groups of segment indices that form connected chains (set by from_text).
        self.stroke_groups: list[list[int]] = []

        n_free = (n_strokes + 1) // 2 if symmetric else n_strokes
        pts = torch.rand(n_free, 4, 2, device=device) * 0.6 + 0.2
        self.control_points = nn.Parameter(pts)

        ys = torch.linspace(0.0, 1.0, size, device=device)
        xs = torch.linspace(0.0, 1.0, size, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer(
            "pixel_grid",
            torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2),
        )

    def all_strokes(self) -> torch.Tensor:
        """
        Return (n_strokes, 4, 2) including auto-generated mirror strokes.

        For non-symmetric glyphs, chain connectivity is enforced: within each
        group in self.stroke_groups, the start of segment[k+1] is fixed to the
        end of segment[k].  This keeps strokes from drifting apart during
        optimisation while remaining fully differentiable.

        Mirror of stroke [p0,p1,p2,p3] is [(1-p3),(1-p2),(1-p1),(1-p0)] —
        a 180° rotation around canvas centre (0.5, 0.5) with reversed
        direction so the stroke reads naturally in both orientations.
        """
        pts = self.control_points.clamp(0.0, 1.0)

        if not self.symmetric:
            # Enforce C0 continuity within each chain group.
            if self.stroke_groups:
                rows = list(pts.unbind(0))        # list of (4, 2)
                for group in self.stroke_groups:
                    for k in range(len(group) - 1):
                        cur, nxt = group[k], group[k + 1]
                        # Override p0 of nxt with p3 of cur (differentiable).
                        rows[nxt] = torch.cat([rows[cur][3:4], rows[nxt][1:]], dim=0)
                pts = torch.stack(rows)
            return pts

        # Mirror: invert coords then flip control-point order
        mirror = (1.0 - pts).flip(1)          # (n_free, 4, 2)

        if self.n_strokes % 2 == 1:
            # Odd total: the last base stroke has no mirror.
            # Force it to be self-symmetric by averaging with its own mirror.
            mid      = ((pts[-1:] + (1.0 - pts[-1:]).flip(1)) * 0.5)
            base     = pts[:-1]
            mirrors  = mirror[:-1]
            return torch.cat([base, mid, mirrors], dim=0)

        return torch.cat([pts, mirror], dim=0)  # (n_strokes, 4, 2)

    def render(self) -> torch.Tensor:
        """Returns (1, H, W) in [0, 1] — white background, black strokes."""
        coverage = _stroke_coverage(
            self.pixel_grid,   # type: ignore[arg-type]
            self.all_strokes(),
            width=self.stroke_width,
        )
        return (1.0 - coverage.reshape(self.size, self.size)).unsqueeze(0)

    # ------------------------------------------------------------------

    @classmethod
    def from_text(
        cls,
        char_a:       str,
        char_b:       str | None   = None,
        n_strokes:    int          = 4,
        size:         int          = 256,
        stroke_width: float        = 0.04,
        device:       torch.device = torch.device("cpu"),
        symmetric:    bool         = False,
    ) -> "BezierGlyph":
        """
        Seed control points from a system font (via font_strokes), falling
        back to hand-crafted skeleton templates in letter_skeletons.py.

        symmetric=True  — only base strokes seeded from char_a; mirrors
                          auto-generate the char_b (rotated) half.
        symmetric=False — n//2 strokes from char_a, n//2 from char_b rotated
                          180°, giving the optimizer a combined starting shape.
        """
        from .letter_skeletons import SKELETONS
        from .font_strokes import get_letter_strokes

        glyph  = cls(n_strokes, size, stroke_width, device, symmetric)
        n_free = glyph.control_points.shape[0]

        def _rot180(stroke: list[tuple[float, float]]) -> list[tuple[float, float]]:
            return [(1.0 - x, 1.0 - y) for x, y in reversed(stroke)]

        def _get(
            letter: str, n: int, flip: bool
        ) -> tuple[list[list[tuple]], list[list[int]]]:
            result = get_letter_strokes(letter.upper(), n)
            if result is not None:
                segs, groups = result
            else:
                raw_sk = SKELETONS.get(letter.upper(), [])
                segs = [raw_sk[i % len(raw_sk)] for i in range(n)] if raw_sk else []
                # Skeletons have no connectivity info — treat as independent.
                groups = [[i] for i in range(len(segs))]

            if flip:
                segs = [_rot180(s) for s in segs]
                # Flipping reverses each stroke's direction, so chain
                # connectivity runs in the opposite order.
                groups = [list(reversed(g)) for g in groups]
            return segs, groups

        if symmetric:
            selected, grps = _get(char_a, n_free, flip=False)
        else:
            # Initialize entirely from char_a — one unified form whose strokes
            # are shared between both orientations.  The optimizer simultaneously
            # satisfies the char_a (upright) and char_b (rotated) classifier
            # signals, deforming the single shape until it reads as both.
            selected, grps = _get(char_a, n_free, flip=False)

        if len(selected) == n_free:
            noise = torch.randn(n_free, 4, 2, device=device) * 0.015
            pts   = torch.tensor(selected, dtype=torch.float32, device=device)
            glyph.control_points = nn.Parameter((pts + noise).clamp(0.0, 1.0))
            # Connectivity only applies to non-symmetric glyphs; symmetric mode
            # enforces its own structure via the mirror computation.
            if not symmetric:
                glyph.stroke_groups = grps

        return glyph

    # ------------------------------------------------------------------

    def to_svg(self, path: str | Path) -> None:
        """Export strokes as an SVG file."""
        s   = self.size
        sw  = max(1.5, self.stroke_width * s)
        pts = self.all_strokes().detach().cpu().numpy() * s

        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{s}" height="{s}" viewBox="0 0 {s} {s}">',
            f'<rect width="{s}" height="{s}" fill="white"/>',
        ]
        for curve in pts:
            p0, p1, p2, p3 = curve
            d = (f"M {p0[0]:.2f},{p0[1]:.2f} "
                 f"C {p1[0]:.2f},{p1[1]:.2f} "
                 f"{p2[0]:.2f},{p2[1]:.2f} "
                 f"{p3[0]:.2f},{p3[1]:.2f}")
            lines.append(
                f'<path d="{d}" fill="none" stroke="black" '
                f'stroke-width="{sw:.2f}" stroke-linecap="round" stroke-linejoin="round"/>'
            )
        lines.append("</svg>")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("\n".join(lines))
