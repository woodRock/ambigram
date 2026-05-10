from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .bezier_glyph import BezierGlyph
from .char_classifier import CharClassifier
from .utils.image import rotate_180


def letter_pairs(word: str) -> list[tuple[str, str]]:
    N, w = len(word), word.upper()
    return [(w[i], w[N - 1 - i]) for i in range((N + 1) // 2)]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Update message sent to the UI
# ---------------------------------------------------------------------------

@dataclass
class OptimUpdate:
    step: int
    phase: str      # "warmstart" | "cmaes" | "done"
    score: float    # 0–1, mean classifier confidence across all pairs × orientations
    pct: float      # 0–100 overall progress (pre-computed so UI doesn't have to)
    svg: str        # full upright word SVG string
    svg_rot: str    # same word rotated 180°


# ---------------------------------------------------------------------------
# SVG composition
# ---------------------------------------------------------------------------

def _glyph_paths_svg(glyph: BezierGlyph) -> str:
    """SVG <path> elements for a single glyph."""
    s   = glyph.size
    sw  = max(1.5, glyph.stroke_width * s)
    pts = glyph.all_strokes().detach().cpu().numpy() * s
    lines = []
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
    return "\n".join(lines)


def compose_svg(glyphs: list[BezierGlyph], N: int) -> tuple[str, str]:
    """
    Return (upright_svg, rotated_svg) for the full N-character word.

    Tile layout mirrors compose() in self_ambigram.py:
      even N=4: [g0 | g1 | rot(g1) | rot(g0)]
      odd  N=5: [g0 | g1 | g2 | rot(g1) | rot(g0)]

    The rotated SVG wraps everything in a single rotate(180) transform —
    mathematically identical to flipping the physical printout.
    """
    if not glyphs:
        return "<svg/>", "<svg/>"

    s = glyphs[0].size
    k = len(glyphs)

    # Build tile list: (glyph_index, apply_rot_to_tile)
    tiles: list[tuple[int, bool]] = [(i, False) for i in range(k)]
    start = k - 2 if N % 2 == 1 else k - 1
    for i in range(start, -1, -1):
        tiles.append((i, True))

    n_tiles  = len(tiles)
    total_w  = n_tiles * s

    def _tile_group(glyph_idx: int, is_rot: bool, x: int) -> str:
        paths    = _glyph_paths_svg(glyphs[glyph_idx])
        rot_attr = f" rotate(180 {s / 2:.1f} {s / 2:.1f})" if is_rot else ""
        return (
            f'<g transform="translate({x} 0){rot_attr}">\n'
            f'  <rect width="{s}" height="{s}" fill="white"/>\n'
            f'{paths}\n'
            f'</g>\n'
        )

    inner = "".join(
        _tile_group(gi, ir, ti * s)
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
                    f'<g transform="rotate(180 {total_w / 2:.1f} {s / 2:.1f})">\n'
                    + inner + "</g>\n</svg>")
        return hdr + inner + "</svg>"

    return _svg(False), _svg(True)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class SVGAmbigramOptimizer:
    """
    Two-phase SVG ambigram optimizer.

    Phase 1 — gradient warm-start:
        Adam on Bezier control points, using the character classifier as the
        sole loss.  No CLIP — the classifier forward pass is ~1 ms on MPS,
        giving interactive-rate updates on an M2 Mac.

    Phase 2 — CMA-ES refinement:
        Black-box evolutionary search.  Each evaluation renders all glyphs,
        runs the classifier in both orientations, and returns the mean
        confidence as the fitness.  CMA-ES explores the global parameter
        space and escapes local minima that gradient descent gets stuck in.

    `on_update` is called from the background thread every few steps with
    an OptimUpdate containing the current SVG strings — wire it up to a
    WebSocket queue for the live demo.
    """

    def __init__(
        self,
        word: str,
        classifier: CharClassifier,
        device: torch.device,
        n_strokes: int = 6,
        glyph_size: int = 128,
        warmstart_steps: int = 300,
        cmaes_budget: int = 2000,
        on_update: Optional[Callable[[OptimUpdate], None]] = None,
    ) -> None:
        self.word             = word.upper()
        self.pairs            = letter_pairs(self.word)
        self.classifier       = classifier
        self.device           = device
        self.glyph_size       = glyph_size
        self.warmstart_steps  = warmstart_steps
        self.cmaes_budget     = cmaes_budget
        self.on_update        = on_update or (lambda _: None)

        self.glyphs: list[BezierGlyph] = [
            BezierGlyph.from_text(
                char_a,
                char_b=char_b,
                n_strokes=n_strokes,
                size=glyph_size,
                stroke_width=0.07,
                device=device,
                symmetric=(char_a == char_b),
            )
            for char_a, char_b in self.pairs
        ]

        # Anchor: remember the template initialization so warm-start can
        # penalize drifting too far from the letter skeleton shapes.
        self._anchors: list[torch.Tensor] = [
            g.control_points.detach().clone() for g in self.glyphs
        ]

    # ------------------------------------------------------------------
    # Scoring (no grad, used by both phases)
    # ------------------------------------------------------------------

    def _smoothness_loss(self, idx: int) -> torch.Tensor:
        """
        Penalise control handles that deviate far from the chord line.

        For a cubic Bézier [p0,c1,c2,p3], the 'natural' handle positions are
        p0+chord/3 and p3-chord/3 (a straight-line degenerate cubic).  Penalising
        departure from those prevents the optimiser finding loop-shaped adversarial
        solutions that fool the classifier but look nothing like letters.
        """
        strokes = self.glyphs[idx].all_strokes()   # (n, 4, 2)
        p0, c1, c2, p3 = strokes[:,0], strokes[:,1], strokes[:,2], strokes[:,3]
        chord   = p3 - p0
        ideal_c1 = p0 + chord / 3.0
        ideal_c2 = p3 - chord / 3.0
        return ((c1 - ideal_c1).pow(2) + (c2 - ideal_c2).pow(2)).mean()

    @torch.no_grad()
    def _score_glyph(self, idx: int) -> float:
        """Classifier confidence for one letter-pair glyph → [0, 1]."""
        char_a, char_b = self.pairs[idx]
        img     = self.glyphs[idx].render()
        img_rot = rotate_180(img)
        p_a = F.softmax(self.classifier(img.unsqueeze(0)),     dim=-1)[
            0, CharClassifier.char_index(char_a)].item()
        p_b = F.softmax(self.classifier(img_rot.unsqueeze(0)), dim=-1)[
            0, CharClassifier.char_index(char_b)].item()
        return (p_a + p_b) / 2

    @torch.no_grad()
    def _score(self) -> float:
        """Mean classifier confidence across all pairs × both orientations → [0, 1]."""
        return sum(self._score_glyph(i) for i in range(len(self.pairs))) / len(self.pairs)

    def _emit(self, step: int, phase: str, score: float, pct: float) -> None:
        svg, svg_rot = compose_svg(self.glyphs, len(self.word))
        self.on_update(OptimUpdate(step=step, phase=phase,
                                   score=score, pct=pct, svg=svg, svg_rot=svg_rot))

    # ------------------------------------------------------------------
    # Per-glyph optimisation (sequential — each pair solved independently)
    # ------------------------------------------------------------------

    def _warmstart_glyph(
        self, idx: int, pct_fn: Callable[[int], float]
    ) -> None:
        """Adam warm-start for a single letter-pair glyph."""
        g              = self.glyphs[idx]
        char_a, char_b = self.pairs[idx]

        opt   = torch.optim.Adam([g.control_points], lr=1.5e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.warmstart_steps, eta_min=1e-3
        )

        for step in range(self.warmstart_steps):
            img  = g.render()
            loss = (self.classifier.readability_loss(img,             char_a) +
                    self.classifier.readability_loss(rotate_180(img), char_b))

            centres = g.control_points.mean(1)
            diff    = centres.unsqueeze(0) - centres.unsqueeze(1)
            loss    = loss + 0.5 * F.relu(0.12 - diff.norm(dim=-1)).mean()

            drift = (g.control_points - self._anchors[idx]).pow(2).mean()
            loss  = loss + 0.4 * drift

            loss = loss + 0.6 * self._smoothness_loss(idx)

            opt.zero_grad(); loss.backward(); opt.step(); sched.step()

            with torch.no_grad():
                g.control_points.data.clamp_(0.0, 1.0)

            if step % 10 == 0 or step == self.warmstart_steps - 1:
                self._emit(step, "warmstart", self._score_glyph(idx), pct_fn(step))

    def _cmaes_glyph(
        self, idx: int, pct_fn: Callable[[int], float]
    ) -> None:
        """CMA-ES refinement for a single letter-pair glyph."""
        try:
            import cma
        except ImportError:
            log.warning("cma not installed — skipping CMA-ES.  pip install cma")
            return

        g = self.glyphs[idx]

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
            score  = self._score_glyph(idx)
            smooth = self._smoothness_loss(idx).item()
            return -(score - 0.3 * smooth)

        x0    = _pack()
        n_dim = len(x0)
        opts  = {
            "maxfevals": self.cmaes_budget,
            "verbose":   -9,
            "bounds":    [0.0, 1.0],
            "popsize":   max(8, 4 + int(3 * np.log(n_dim))),
        }

        es         = cma.CMAEvolutionStrategy(x0, 0.04, opts)
        best_score = -_fitness(x0)
        best_x     = x0.copy()
        eval_count = 0
        popsize    = opts["popsize"]

        while not es.stop():
            candidates = es.ask()
            fitnesses  = [_fitness(x) for x in candidates]
            es.tell(candidates, fitnesses)
            eval_count += len(candidates)

            gen_best = min(fitnesses)
            if -gen_best > best_score or eval_count % (popsize * 3) == 0:
                if -gen_best > best_score:
                    best_score = -gen_best
                    best_x     = candidates[fitnesses.index(gen_best)].copy()
                _unpack(best_x)
                self._emit(eval_count, "cmaes", best_score, pct_fn(eval_count))

        _unpack(best_x)

    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Optimise all glyphs in parallel — each letter pair runs its own
        warm-start → CMA-ES cycle independently, composing the full word
        SVG on every update so the UI sees all letters evolving together.
        """
        n          = len(self.pairs)
        per_glyph  = self.warmstart_steps + self.cmaes_budget
        glyph_step = [0] * n          # steps completed per glyph
        lock       = threading.Lock()

        def _pct(glyph_idx: int, local_step: int) -> float:
            with lock:
                glyph_step[glyph_idx] = local_step
                return sum(glyph_step) / (n * per_glyph) * 100

        def _run_one(i: int) -> None:
            a, b = self.pairs[i]
            log.info("Glyph %d/%d  %s↔%s  warm-start", i + 1, n, a, b)
            self._warmstart_glyph(i, lambda step: _pct(i, step))
            log.info("Glyph %d/%d  %s↔%s  CMA-ES", i + 1, n, a, b)
            self._cmaes_glyph(i, lambda step: _pct(i, self.warmstart_steps + step))

        threads = [
            threading.Thread(target=_run_one, args=(i,), daemon=True)
            for i in range(n)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self._emit(n * per_glyph, "done", self._score(), 100.0)
