from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bezier_glyph import BezierGlyph
from .char_classifier import CharClassifier


def letter_pairs(word: str) -> list[tuple[str, str]]:
    N, w = len(word), word.upper()
    return [(w[i], w[N - 1 - i]) for i in range((N + 1) // 2)]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Update message sent to the UI
# ---------------------------------------------------------------------------

@dataclass
class OptimUpdate:
    step:    int
    phase:   str    # "warmstart" | "cmaes" | "done"
    score:   float  # 0–1 mean classifier confidence
    pct:     float  # 0–100 overall progress
    svg:     str
    svg_rot: str


# ---------------------------------------------------------------------------
# SVG composition  (supports variable-width glyphs / bigrams)
# ---------------------------------------------------------------------------

def compose_svg(glyphs: list[BezierGlyph], N: int) -> tuple[str, str]:
    """
    Tile glyphs into a full word SVG.  Glyphs may have width_tiles > 1
    (bigrams) — the tiling logic automatically accounts for variable widths.

    Layout mirrors the original per-pair scheme:
      tiles = [g0, g1, …, g_{k-1},  mirror(g_{start}), …, mirror(g_0)]
    where start = k-2 for odd N, k-1 for even N.
    """
    if not glyphs:
        return "<svg/>", "<svg/>"

    s = glyphs[0].size
    k = len(glyphs)

    tiles: list[tuple[int, bool]] = [(i, False) for i in range(k)]
    start = k - 2 if N % 2 == 1 else k - 1
    for i in range(start, -1, -1):
        tiles.append((i, True))

    # x offset for each tile, accounting for variable widths.
    x_offsets: list[int] = []
    x_cur = 0
    for gi, _ in tiles:
        x_offsets.append(x_cur)
        x_cur += glyphs[gi].width_tiles * s
    total_w = x_cur

    def _tile_group(gi: int, is_rot: bool, x: int) -> str:
        g  = glyphs[gi]
        tw = g.width_tiles * s
        sw = max(1.5, g.stroke_width * s)
        pts = g.all_strokes().detach().cpu().numpy() * np.array([tw, s])

        path_lines = []
        for curve in pts:
            p0, p1, p2, p3 = curve
            d = (f"M {p0[0]:.2f},{p0[1]:.2f} "
                 f"C {p1[0]:.2f},{p1[1]:.2f} "
                 f"{p2[0]:.2f},{p2[1]:.2f} "
                 f"{p3[0]:.2f},{p3[1]:.2f}")
            path_lines.append(
                f'<path d="{d}" fill="none" stroke="black" '
                f'stroke-width="{sw:.2f}" stroke-linecap="round" stroke-linejoin="round"/>'
            )
        paths    = "\n".join(path_lines)
        rot_attr = f" rotate(180 {tw/2:.1f} {s/2:.1f})" if is_rot else ""
        return (
            f'<g transform="translate({x} 0){rot_attr}">\n'
            f'  <rect width="{tw}" height="{s}" fill="white"/>\n'
            f'{paths}\n'
            f'</g>\n'
        )

    inner = "".join(
        _tile_group(gi, ir, x_offsets[ti])
        for ti, (gi, ir) in enumerate(tiles)
    )

    def _svg(flip_whole: bool) -> str:
        hdr = (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{total_w}" height="{s}" '
            f'viewBox="0 0 {total_w} {s}">\n'
            f'<rect width="{total_w}" height="{s}" fill="white"/>\n'
        )
        if flip_whole:
            return (hdr +
                    f'<g transform="rotate(180 {total_w/2:.1f} {s/2:.1f})">\n' +
                    inner + "</g>\n</svg>")
        return hdr + inner + "</svg>"

    return _svg(False), _svg(True)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class SVGAmbigramOptimizer:
    """
    Per-glyph ambigram optimizer with + initialisation, bigram support,
    and pixel-MSE readability loss against Hershey letter templates.

    Architecture
    ------------
    Adjacent non-symmetric letter pairs are merged into *bigram* glyphs
    (width_tiles=2).  Each bigram has 2× the strokes and can use curves
    that cross the letter boundary — the key visual feature of hand-drawn
    ambigrams.

    Symmetric self-pairs (S↔S, N↔N, …) skip optimisation entirely; the
    Hershey initialisation is already a correct ambigram for those.

    All optimisable glyphs start from a neutral asterisk (+ / *) shape —
    a 180°-symmetric blank slate that gives the optimiser equal freedom to
    deform toward either target letter.

    Loss
    ----
    Pixel-MSE to Hershey-rendered letter templates replaces cross-entropy
    from the classifier.  MSE provides dense, pixel-level gradient even when
    the current shape bears no resemblance to a letter — exactly the regime
    that killed the classifier-based approach.

    Smoothness and stroke-separation penalties are retained unchanged.
    """

    def __init__(
        self,
        word:            str,
        classifier:      CharClassifier,
        device:          torch.device,
        n_strokes:       int  = 6,
        glyph_size:      int  = 128,
        warmstart_steps: int  = 300,
        cmaes_budget:    int  = 2000,
        on_update: Optional[Callable[[OptimUpdate], None]] = None,
    ) -> None:
        self.word            = word.upper()
        self.pairs           = letter_pairs(self.word)
        self.classifier      = classifier
        self.device          = device
        self.glyph_size      = glyph_size
        self.warmstart_steps = warmstart_steps
        self.cmaes_budget    = cmaes_budget
        self.on_update       = on_update or (lambda _: None)

        # ── Build glyph groups ──────────────────────────────────────────
        # Merge adjacent non-symmetric pairs into bigrams.
        K      = len(self.pairs)
        groups: list[list[int]] = []
        i = 0
        while i < K:
            a, b = self.pairs[i]
            if a == b:
                # Symmetric self-pair — never merged.
                groups.append([i])
                i += 1
            elif i + 1 < K and self.pairs[i + 1][0] != self.pairs[i + 1][1]:
                # Two adjacent non-symmetric pairs → bigram.
                groups.append([i, i + 1])
                i += 2
            else:
                groups.append([i])
                i += 1
        self.groups = groups

        # ── Create glyphs ───────────────────────────────────────────────
        self.glyphs: list[BezierGlyph] = []
        for group in groups:
            width   = len(group)
            a0, b0  = self.pairs[group[0]]
            is_sym  = (a0 == b0) and width == 1
            if is_sym:
                # Hershey init — this glyph is already a valid ambigram.
                self.glyphs.append(BezierGlyph.from_text(
                    char_a=a0, char_b=b0,
                    n_strokes=n_strokes, size=glyph_size,
                    stroke_width=0.07, device=device, symmetric=True,
                ))
            else:
                # Hershey-seeded init: place each letter's strokes in their
                # tile slice.  Falls back to + init if Hershey unavailable.
                self.glyphs.append(self._make_bigram_glyph(
                    group, n_strokes, glyph_size, device,
                ))

        # ── Pre-render Hershey templates for pixel-MSE loss ─────────────
        self._templates: dict[str, torch.Tensor] = self._build_templates(
            n_strokes=max(n_strokes, 6)
        )
        # All-white baseline MSE per glyph — used to normalise score to [0,1].
        self._baselines: list[float] = self._compute_baselines()

    # ------------------------------------------------------------------
    # Bigram glyph factory
    # ------------------------------------------------------------------

    def _make_bigram_glyph(
        self,
        group:      list[int],
        n_strokes:  int,
        glyph_size: int,
        device:     torch.device,
    ) -> BezierGlyph:
        """
        Initialise a bigram glyph with Hershey letter strokes scaled into
        each tile slice.  Using actual letter shapes as a starting point
        dramatically reduces the distance the optimiser needs to travel
        compared to the + / asterisk seed.

        For each tile k in [0, width]:
          - Take the *upright* letter (a) Hershey strokes
          - Scale x from [0,1] → [k/width, (k+1)/width] in bigram canvas coords

        The optimiser then deforms these into a compromise shape that also
        satisfies the rotated-view readability constraint.

        Falls back to BezierGlyph.from_plus() if HersheyFonts is unavailable.
        """
        from .font_strokes import get_letter_strokes

        width = len(group)
        all_pts: list[list] = []

        for k, pair_idx in enumerate(group):
            a, _ = self.pairs[pair_idx]
            x0   = k / width
            x1   = (k + 1) / width

            result = get_letter_strokes(a, n_strokes)
            if result is not None:
                segs, _ = result
                for seg in segs:
                    scaled = [(x0 + x * (x1 - x0), y) for (x, y) in seg]
                    all_pts.append(scaled)
            else:
                # No Hershey — fall back to + arms for this tile
                plus = BezierGlyph.from_plus(
                    n_strokes=n_strokes, size=glyph_size,
                    stroke_width=0.07 / width, device=device,
                    symmetric=False, width_tiles=1,
                )
                sub = plus.control_points.detach().clone()
                sub[:, :, 0] = x0 + sub[:, :, 0] * (x1 - x0)
                all_pts.extend(sub.numpy().tolist())

        noise = torch.randn(len(all_pts), 4, 2, device=device) * 0.015
        pts   = torch.tensor(all_pts, dtype=torch.float32, device=device)

        glyph = BezierGlyph(
            n_strokes=n_strokes * width,
            size=glyph_size,
            stroke_width=0.07 / width,
            device=device,
            symmetric=False,
            width_tiles=width,
        )
        glyph.control_points = nn.Parameter((pts + noise).clamp(0.01, 0.99))
        return glyph

    # ------------------------------------------------------------------
    # Template rendering
    # ------------------------------------------------------------------

    def _build_templates(self, n_strokes: int) -> dict[str, torch.Tensor]:
        """Render each unique letter via Hershey init — these are the MSE targets."""
        unique = {c for pair in self.pairs for c in pair}
        out: dict[str, torch.Tensor] = {}
        for char in unique:
            g = BezierGlyph.from_text(
                char_a=char, n_strokes=n_strokes,
                size=self.glyph_size, stroke_width=0.07,
                device=self.device, symmetric=False,
            )
            with torch.no_grad():
                out[char] = g.render().detach()   # (1, s, s)
        return out

    # ------------------------------------------------------------------
    # Baseline (all-white prediction MSE) for score normalisation
    # ------------------------------------------------------------------

    def _compute_baselines(self) -> list[float]:
        s = self.glyph_size
        baselines: list[float] = []
        for gi, group in enumerate(self.groups):
            W    = len(group)
            g    = self.glyphs[gi]
            white = torch.ones(1, s, g.width_tiles * s, device=self.device)
            rot   = white.flip([-1, -2])
            b     = 0.0
            for k, pair_idx in enumerate(group):
                ch_a, ch_b = self.pairs[pair_idx]
                crop   = white[:, :,  k*s : (k+1)*s]
                crop_r = rot[:, :, (W-1-k)*s : (W-k)*s]
                b += F.mse_loss(crop,   self._templates[ch_a]).item()
                b += F.mse_loss(crop_r, self._templates[ch_b]).item()
            baselines.append(max(b, 1e-8))
        return baselines

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def _glyph_loss(self, gi: int) -> torch.Tensor:
        """
        Pixel-MSE readability loss for glyph gi.

        Single glyph (width=1): MSE(img, tmpl_a) + MSE(rotate180(img), tmpl_b)
        Bigram (width=2):       four MSE terms — each half upright + rotated.
        """
        g     = self.glyphs[gi]
        group = self.groups[gi]
        s     = self.glyph_size
        img   = g.render()              # (1, s, width*s)
        rot   = img.flip([-1, -2])      # rotate 180°

        loss = torch.tensor(0.0, device=self.device)
        W    = len(group)
        for k, pair_idx in enumerate(group):
            a, b    = self.pairs[pair_idx]
            crop    = img[:, :,  k*s : (k+1)*s]
            # After 180° flip the crop order reverses: index W-1-k
            crop_r  = rot[:, :, (W-1-k)*s : (W-k)*s]
            loss    = loss + F.mse_loss(crop,   self._templates[a])
            loss    = loss + F.mse_loss(crop_r, self._templates[b])
        return loss

    def _smoothness_loss(self, gi: int) -> torch.Tensor:
        strokes = self.glyphs[gi].all_strokes()
        p0, c1, c2, p3 = strokes[:,0], strokes[:,1], strokes[:,2], strokes[:,3]
        chord = p3 - p0
        return (((c1 - (p0 + chord/3)).pow(2) + (c2 - (p3 - chord/3)).pow(2)).mean())

    # ------------------------------------------------------------------
    # Scoring — MSE-based: 0 = no better than blank, 1 = perfect match
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _score_glyph(self, gi: int) -> float:
        loss     = self._glyph_loss(gi).item()
        baseline = self._baselines[gi]
        return max(0.0, 1.0 - loss / baseline)

    @torch.no_grad()
    def _score(self) -> float:
        total, n = 0.0, 0
        for gi, group in enumerate(self.groups):
            total += self._score_glyph(gi) * len(group)
            n     += len(group)
        return total / n if n > 0 else 0.0

    def _emit(self, step: int, phase: str, score: float, pct: float) -> None:
        svg, svg_rot = compose_svg(self.glyphs, len(self.word))
        self.on_update(OptimUpdate(
            step=step, phase=phase, score=score, pct=pct,
            svg=svg, svg_rot=svg_rot,
        ))

    # ------------------------------------------------------------------
    # Per-glyph optimisation
    # ------------------------------------------------------------------

    def _warmstart_glyph(self, gi: int, pct_fn: Callable[[int], float]) -> None:
        g = self.glyphs[gi]

        opt   = torch.optim.Adam([g.control_points], lr=2e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.warmstart_steps, eta_min=1e-3
        )

        for step in range(self.warmstart_steps):
            loss = self._glyph_loss(gi)
            loss = loss + 0.4 * self._smoothness_loss(gi)

            # Repel strokes from each other to prevent collapse.
            centres = g.control_points.mean(1)
            diff    = centres.unsqueeze(0) - centres.unsqueeze(1)
            loss    = loss + 0.3 * F.relu(0.12 - diff.norm(dim=-1)).mean()

            opt.zero_grad(); loss.backward(); opt.step(); sched.step()

            with torch.no_grad():
                g.control_points.data.clamp_(0.0, 1.0)

            if step % 10 == 0 or step == self.warmstart_steps - 1:
                self._emit(step, "warmstart", self._score_glyph(gi), pct_fn(step))

    def _cmaes_glyph(self, gi: int, pct_fn: Callable[[int], float]) -> None:
        try:
            import cma
        except ImportError:
            log.warning("cma not installed — skipping CMA-ES.  pip install cma")
            return

        g = self.glyphs[gi]

        def _pack() -> np.ndarray:
            return g.control_points.detach().cpu().numpy().flatten()

        def _unpack(x: np.ndarray) -> None:
            pts = (torch.from_numpy(x.astype(np.float32))
                   .reshape(g.control_points.shape)
                   .clamp(0.0, 1.0)
                   .to(self.device))
            g.control_points.data.copy_(pts)

        def _fitness(x: np.ndarray) -> float:
            _unpack(x)
            # MSE-based fitness: minimise (lower = more letter-like).
            loss   = self._glyph_loss(gi).item()
            smooth = self._smoothness_loss(gi).item()
            return loss + 0.3 * smooth

        x0    = _pack()
        n_dim = len(x0)
        opts  = {
            "maxfevals": self.cmaes_budget,
            "verbose":   -9,
            "bounds":    [0.0, 1.0],
            "popsize":   max(8, 4 + int(3 * np.log(n_dim))),
        }

        es           = cma.CMAEvolutionStrategy(x0, 0.05, opts)
        best_fitness = _fitness(x0)
        best_x       = x0.copy()
        eval_count   = 0
        popsize      = opts["popsize"]

        while not es.stop():
            candidates = es.ask()
            fitnesses  = [_fitness(x) for x in candidates]
            es.tell(candidates, fitnesses)
            eval_count += len(candidates)

            gen_best = min(fitnesses)
            if gen_best < best_fitness or eval_count % (popsize * 3) == 0:
                if gen_best < best_fitness:
                    best_fitness = gen_best
                    best_x       = candidates[fitnesses.index(gen_best)].copy()
                _unpack(best_x)
                self._emit(eval_count, "cmaes", self._score_glyph(gi), pct_fn(eval_count))

        _unpack(best_x)

    # ------------------------------------------------------------------

    def run(self) -> None:
        """Optimise all glyphs in parallel — skip symmetric self-pairs."""
        n   = len(self.glyphs)
        per = self.warmstart_steps + self.cmaes_budget

        non_sym = sum(
            1 for gi, group in enumerate(self.groups)
            if not (len(group) == 1 and self.pairs[group[0]][0] == self.pairs[group[0]][1])
        )
        total_steps = max(non_sym * per, 1)

        glyph_step = [0] * n
        lock       = threading.Lock()

        def _pct(gi: int, step: int) -> float:
            with lock:
                glyph_step[gi] = step
                return sum(glyph_step) / total_steps * 100

        def _run_one(gi: int) -> None:
            group  = self.groups[gi]
            a0, b0 = self.pairs[group[0]]
            if len(group) == 1 and a0 == b0:
                log.info("Glyph %d/%d  %s  symmetric — skipping", gi+1, n, a0)
                return
            label = "+".join(f"{self.pairs[p][0]}↔{self.pairs[p][1]}" for p in group)
            log.info("Glyph %d/%d  [%s]  warm-start", gi+1, n, label)
            self._warmstart_glyph(gi, lambda step: _pct(gi, step))
            log.info("Glyph %d/%d  [%s]  CMA-ES", gi+1, n, label)
            self._cmaes_glyph(gi, lambda step: _pct(gi, per // 2 + step))

        threads = [
            threading.Thread(target=_run_one, args=(gi,), daemon=True)
            for gi in range(n)
        ]
        for t in threads: t.start()
        for t in threads: t.join()

        self._emit(n * per, "done", self._score(), 100.0)
