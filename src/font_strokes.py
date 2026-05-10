"""
Single-stroke letter initializations from Hershey fonts.

Hershey fonts are single-stroke centerline fonts (designed for pen plotters),
so every letter is a list of polyline strokes — no filled outlines.
We convert those polylines to smooth cubic Bézier segments via arc-length
parameterization + C1-continuous tangents.

Falls back to letter_skeletons.py if HersheyFonts is unavailable.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# HersheyFonts import — optional dependency
# ---------------------------------------------------------------------------

try:
    from HersheyFonts import HersheyFonts as _HF  # type: ignore
    _HAVE_HERSHEY = True
except ImportError:
    _HAVE_HERSHEY = False

_HERSHEY: Optional[object] = None


def _load_hershey():
    global _HERSHEY
    if _HERSHEY is not None:
        return _HERSHEY
    if not _HAVE_HERSHEY:
        return None
    h = _HF()
    h.load_default_font("futural")  # clean sans-serif single-stroke
    _HERSHEY = h
    return h


# ---------------------------------------------------------------------------
# Polyline → smooth cubic Bézier segments
# ---------------------------------------------------------------------------

def _polyline_to_beziers(
    pts: list[tuple[float, float]],
    n_segs: int,
) -> list[list[tuple[float, float]]]:
    """
    Convert a polyline to n_segs smooth cubic Bézier segments.

    Arc-length parameterization ensures segments are equal-length.
    Tangents are computed from neighbouring key-points for C1 continuity.
    """
    arr = np.array(pts, dtype=float)
    m = len(arr)
    if m < 2 or n_segs <= 0:
        return []

    diffs    = np.diff(arr, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum      = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total    = cum[-1]
    if total < 1e-8:
        return []

    # n_segs + 1 key points at equal arc-length intervals
    t_vals  = np.linspace(0.0, total, n_segs + 1)
    kpts: list[np.ndarray] = []
    for t in t_vals:
        idx = int(np.searchsorted(cum, t, side="right")) - 1
        idx = int(np.clip(idx, 0, m - 2))
        lt  = float(np.clip((t - cum[idx]) / (seg_lens[idx] + 1e-10), 0.0, 1.0))
        kpts.append(arr[idx] + lt * diffs[idx])

    # Tangent at each key point (central difference, normalised)
    tans: list[np.ndarray] = []
    for i, kp in enumerate(kpts):
        if i == 0:
            raw = kpts[1] - kpts[0]
        elif i == len(kpts) - 1:
            raw = kpts[-1] - kpts[-2]
        else:
            raw = kpts[i + 1] - kpts[i - 1]
        n = float(np.linalg.norm(raw))
        tans.append(raw / n if n > 1e-8 else np.array([1.0, 0.0]))

    seg_step = total / n_segs
    beziers: list[list[tuple[float, float]]] = []
    for i in range(n_segs):
        p0, p3 = kpts[i], kpts[i + 1]
        c1 = p0 + tans[i]     * seg_step / 3
        c2 = p3 - tans[i + 1] * seg_step / 3
        beziers.append([
            (float(p0[0]), float(p0[1])),
            (float(c1[0]), float(c1[1])),
            (float(c2[0]), float(c2[1])),
            (float(p3[0]), float(p3[1])),
        ])
    return beziers


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_letter_strokes(
    letter: str,
    n: int,
) -> Optional[tuple[list[list[tuple[float, float]]], list[list[int]]]]:
    """
    Return (beziers, groups) for *letter* in [0,1] canvas coords.

    beziers : n cubic Bézier strokes as [[p0,c1,c2,p3], ...]
    groups  : list of index-lists; segments within a group are consecutive
              sections of the same Hershey polyline and should stay connected
              end-to-end.  e.g. J → [[0,1,2,3,4,5]],
                                 E → [[0,1],[2,3],[4],[5]]

    Returns None if HersheyFonts is unavailable.
    """
    h = _load_hershey()
    if h is None:
        return None

    raw = list(h.strokes_for_text(letter.upper()))
    if not raw:
        return None

    # ── Normalise all points to [margin, 1-margin] canvas space ──────────
    all_pts = [pt for stroke in raw for pt in stroke]
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    xmin, xmax = float(min(xs)), float(max(xs))
    ymin, ymax = float(min(ys)), float(max(ys))
    w     = (xmax - xmin) or 1.0
    h_sz  = (ymax - ymin) or 1.0
    scale = max(w, h_sz)

    margin = 0.075
    inner  = 1.0 - 2 * margin
    # Centering offsets for non-square glyphs
    cx = (1.0 - w   / scale) * inner / 2
    cy = (1.0 - h_sz / scale) * inner / 2

    def _norm(pt: tuple) -> tuple[float, float]:
        # Hershey Y increases downward already — no flip needed.
        nx = (pt[0] - xmin) / scale * inner + margin + cx
        ny = (pt[1] - ymin) / scale * inner + margin + cy
        return (float(np.clip(nx, 0.0, 1.0)), float(np.clip(ny, 0.0, 1.0)))

    norm_strokes = [[_norm(pt) for pt in stroke] for stroke in raw]

    # ── Distribute n Bézier segments proportionally across strokes ────────
    def _arc(pts: list) -> float:
        a = np.array(pts, dtype=float)
        return float(np.linalg.norm(np.diff(a, axis=0), axis=1).sum())

    lengths = [_arc(s) for s in norm_strokes]
    total   = sum(lengths) or 1.0
    ns      = len(norm_strokes)

    # Decide how many Bézier segments each Hershey stroke gets.
    if n >= ns:
        # At least 1 per stroke; give extras to the longest strokes.
        alloc = [1] * ns
        for _ in range(n - ns):
            ratios = [lengths[j] / alloc[j] for j in range(ns)]
            alloc[int(np.argmax(ratios))] += 1
    else:
        # Fewer segments than strokes: keep only the n longest strokes.
        order = sorted(range(ns), key=lambda j: lengths[j], reverse=True)
        keep  = set(order[:n])
        norm_strokes = [norm_strokes[j] for j in range(ns) if j in keep]
        lengths      = [lengths[j]      for j in range(ns) if j in keep]
        ns    = len(norm_strokes)
        alloc = [1] * ns

    # ── Convert each stroke to Bézier segments, tracking groups ─────────
    beziers: list[list[tuple[float, float]]] = []
    groups:  list[list[int]] = []

    for stroke, k in zip(norm_strokes, alloc):
        segs = _polyline_to_beziers(stroke, k)
        if segs:
            start = len(beziers)
            beziers.extend(segs)
            groups.append(list(range(start, start + len(segs))))

    if not beziers:
        return None

    # Pad to exactly n with repeated singletons if any stroke was degenerate.
    while len(beziers) < n:
        beziers.append(beziers[len(beziers) % len(beziers)])

    return beziers[:n], groups
