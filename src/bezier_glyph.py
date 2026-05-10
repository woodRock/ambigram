from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# SVG tapered stroke utility
# ---------------------------------------------------------------------------

def tapered_path_svg(curve: np.ndarray, max_sw: float, n_samples: int = 24) -> str:
    p0, p1, p2, p3 = curve
    ts = np.linspace(0.0, 1.0, n_samples)
    mt = 1.0 - ts
    bx = mt**3*p0[0] + 3*mt**2*ts*p1[0] + 3*mt*ts**2*p2[0] + ts**3*p3[0]
    by = mt**3*p0[1] + 3*mt**2*ts*p1[1] + 3*mt*ts**2*p2[1] + ts**3*p3[1]
    dx = 3*(mt**2*(p1[0]-p0[0]) + 2*mt*ts*(p2[0]-p1[0]) + ts**2*(p3[0]-p2[0]))
    dy = 3*(mt**2*(p1[1]-p0[1]) + 2*mt*ts*(p2[1]-p1[1]) + ts**2*(p3[1]-p2[1]))
    norms = np.sqrt(dx**2 + dy**2) + 1e-8
    px, py = -dy / norms, dx / norms
    w   = max_sw * np.sin(np.pi * ts) + 0.5
    ux, uy = bx + px * w, by + py * w
    lx, ly = bx - px * w, by - py * w
    pts = list(zip(ux, uy)) + list(zip(lx[::-1], ly[::-1]))
    d   = f"M {pts[0][0]:.1f},{pts[0][1]:.1f}" + "".join(f" L {x:.1f},{y:.1f}" for x, y in pts[1:]) + " Z"
    return f'<path d="{d}" fill="black" stroke="none"/>'


# ---------------------------------------------------------------------------
# Differentiable Bézier rasteriser
# ---------------------------------------------------------------------------

def _bezier3_samples(curves: torch.Tensor, n_samples: int = 32) -> torch.Tensor:
    ts = torch.linspace(0.0, 1.0, n_samples, device=curves.device)
    t  = ts[None, :, None]
    mt = 1.0 - t
    p  = [curves[:, k:k+1, :] for k in range(4)]
    return mt**3*p[0] + 3*mt**2*t*p[1] + 3*mt*t**2*p[2] + t**3*p[3]


def _stroke_coverage(
    px:         torch.Tensor,
    curves:     torch.Tensor,
    width:      float,
    n_samples:  int  = 32,
    chunk_size: int  = 2048,
) -> torch.Tensor:
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
# + / asterisk initialisation helper
# ---------------------------------------------------------------------------

def _make_plus_strokes(
    n_free:      int,
    width_tiles: int,
    symmetric:   bool,
    device:      torch.device,
) -> torch.Tensor:
    """
    n_free Bézier strokes arranged as radiating arms from the centre of each
    tile — a neutral, 180°-symmetric starting shape.

    symmetric=True  → arms span [0, π]; mirror strokes in all_strokes() fill
                       the other hemisphere, giving a full asterisk.
    symmetric=False → arms span [0, 2π] for full coverage from the start.
    """
    pts_list: list = []
    base_per = n_free // width_tiles
    extra    = n_free % width_tiles

    for tile in range(width_tiles):
        cx    = (tile + 0.5) / width_tiles
        cy    = 0.5
        reach = 0.34 / width_tiles   # arm length scales to tile width

        n_here = base_per + (1 if tile < extra else 0)

        for k in range(n_here):
            span  = np.pi if symmetric else 2.0 * np.pi
            angle = span * k / max(n_here, 1)
            ex    = float(np.clip(cx + reach * np.cos(angle), 0.02, 0.98))
            ey    = float(np.clip(cy - reach * np.sin(angle), 0.02, 0.98))

            dx, dy = ex - cx, ey - cy
            p0 = (cx,          cy         )
            c1 = (cx + dx/3.0, cy + dy/3.0)
            c2 = (cx + 2*dx/3, cy + 2*dy/3)
            p3 = (ex,          ey         )
            pts_list.append([p0, c1, c2, p3])

    return torch.tensor(pts_list, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# BezierGlyph
# ---------------------------------------------------------------------------

class BezierGlyph(nn.Module):
    """
    A glyph parameterised as cubic Bézier strokes.

    width_tiles > 1  creates a wide canvas (bigram, word-level, etc.).
    Control points are always in [0,1] × [0,1] of the full canvas.

    symmetric=True   n_strokes//2 base strokes stored; mirrors auto-generated
                     as (1−pts).flip(1) — 180° rotation around canvas centre.
    symmetric=False  all n_strokes are free parameters.
    """

    def __init__(
        self,
        n_strokes:    int          = 4,
        size:         int          = 256,
        stroke_width: float        = 0.04,
        device:       torch.device = torch.device("cpu"),
        symmetric:    bool         = False,
        width_tiles:  int          = 1,
    ) -> None:
        super().__init__()
        self.size         = size
        self.stroke_width = stroke_width
        self.symmetric    = symmetric
        self.n_strokes    = n_strokes
        self.width_tiles  = width_tiles
        self.stroke_groups: list[list[int]] = []

        n_free = (n_strokes + 1) // 2 if symmetric else n_strokes
        pts = torch.rand(n_free, 4, 2, device=device) * 0.6 + 0.2
        self.control_points = nn.Parameter(pts)

        ys = torch.linspace(0.0, 1.0, size, device=device)
        xs = torch.linspace(0.0, 1.0, size * width_tiles, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer(
            "pixel_grid",
            torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2),
        )

    def all_strokes(self) -> torch.Tensor:
        pts = self.control_points.clamp(0.0, 1.0)

        if not self.symmetric:
            if self.stroke_groups:
                rows = list(pts.unbind(0))
                for group in self.stroke_groups:
                    for k in range(len(group) - 1):
                        cur, nxt = group[k], group[k + 1]
                        rows[nxt] = torch.cat([rows[cur][3:4], rows[nxt][1:]], dim=0)
                pts = torch.stack(rows)
            return pts

        mirror = (1.0 - pts).flip(1)
        if self.n_strokes % 2 == 1:
            mid     = (pts[-1:] + (1.0 - pts[-1:]).flip(1)) * 0.5
            return torch.cat([pts[:-1], mid, mirror[:-1]], dim=0)
        return torch.cat([pts, mirror], dim=0)

    def render(self) -> torch.Tensor:
        """Returns (1, H, W) — white background, black strokes."""
        coverage = _stroke_coverage(
            self.pixel_grid,   # type: ignore[arg-type]
            self.all_strokes(),
            width=self.stroke_width,
        )
        return (1.0 - coverage.reshape(self.size, self.size * self.width_tiles)).unsqueeze(0)

    # ------------------------------------------------------------------
    # Initialisers
    # ------------------------------------------------------------------

    @classmethod
    def from_plus(
        cls,
        n_strokes:    int          = 6,
        size:         int          = 128,
        stroke_width: float        = 0.07,
        device:       torch.device = torch.device("cpu"),
        symmetric:    bool         = False,
        width_tiles:  int          = 1,
    ) -> "BezierGlyph":
        """
        Initialise with a neutral asterisk / + shape — 180°-symmetric,
        maximally spread, and a blank slate for the optimiser to deform.
        """
        glyph  = cls(n_strokes, size, stroke_width, device, symmetric, width_tiles)
        n_free = glyph.control_points.shape[0]
        pts    = _make_plus_strokes(n_free, width_tiles, symmetric, device)
        noise  = torch.randn_like(pts) * 0.01
        glyph.control_points = nn.Parameter((pts + noise).clamp(0.02, 0.98))
        return glyph

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
        Seed control points from Hershey fonts.  Used for symmetric self-pairs
        and for rendering letter templates (pixel-MSE targets).
        """
        from .letter_skeletons import SKELETONS
        from .font_strokes import get_letter_strokes

        glyph  = cls(n_strokes, size, stroke_width, device, symmetric)
        n_free = glyph.control_points.shape[0]

        def _rot180(stroke):
            return [(1.0 - x, 1.0 - y) for x, y in reversed(stroke)]

        def _get(letter, n, flip):
            result = get_letter_strokes(letter.upper(), n)
            if result is not None:
                segs, groups = result
            else:
                raw    = SKELETONS.get(letter.upper(), [])
                segs   = [raw[i % len(raw)] for i in range(n)] if raw else []
                groups = [[i] for i in range(len(segs))]
            if flip:
                segs   = [_rot180(s) for s in segs]
                groups = [list(reversed(g)) for g in groups]
            return segs, groups

        if symmetric:
            selected, grps = _get(char_a, n_free, flip=False)
        else:
            segs_a, grps_a = _get(char_a, n_free, flip=False)
            other = char_b or char_a
            if other != char_a:
                segs_b, grps_b = _get(other, n_free, flip=False)
                selected, grps = (segs_b, grps_b) if len(grps_b) < len(grps_a) else (segs_a, grps_a)
            else:
                selected, grps = segs_a, grps_a

        if len(selected) == n_free:
            noise = torch.randn(n_free, 4, 2, device=device) * 0.015
            pts   = torch.tensor(selected, dtype=torch.float32, device=device)
            glyph.control_points = nn.Parameter((pts + noise).clamp(0.0, 1.0))
            if not symmetric:
                glyph.stroke_groups = grps

        return glyph

    # ------------------------------------------------------------------

    def to_svg(self, path: str | Path) -> None:
        s       = self.size
        total_w = s * self.width_tiles
        sw      = max(1.5, self.stroke_width * s)
        pts     = self.all_strokes().detach().cpu().numpy() * np.array([total_w, s])

        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{total_w}" height="{s}" viewBox="0 0 {total_w} {s}">',
            f'<rect width="{total_w}" height="{s}" fill="white"/>',
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
