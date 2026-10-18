"""
What every modality returns, and the two ways of drawing evidence onto a curve.

The shared vocabulary of the segmentation package — deliberately the only thing
the three modality modules have in common.

The curves follow "Evidenzkurven: Breite τ und Gipfel": a hint at a point in
time becomes a Gaussian bell with σ = τ, a spoken sentence a raised-cosine
window; several entries are merged by pointwise maximum, and every curve is
sampled on the GRID_HZ grid E[n] = E(n / GRID_HZ).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from docupilot.config import GRID_HZ

__all__ = [
    "BOUNDARY_THRESHOLD", "GRID_HZ", "BoundaryEvidence",
    "draw_gaussian", "draw_window", "grid",
]

# At or above this probability a modality commits to a boundary.
BOUNDARY_THRESHOLD = 0.5

# Beyond 3σ a bell is below 1.2 % of its height and is cut off.
_GAUSS_CUTOFF_SIGMA = 3.0


@dataclass(frozen=True)
class BoundaryEvidence:
    """
    One modality's answer to "where are the action boundaries?": a graded curve
    for display and for the rule decider, the boundaries the modality commits
    to, and — when the modality measures several things — its sub-curves.

    The sub-curves share `times_s`. The trained deciders read them as separate
    feature blocks; all of them belong to this one modality, so the Shapley
    decomposition stays one over three players.
    """

    times_s: np.ndarray          # (T,) float64 — timestamp per sample
    score: np.ndarray            # (T,) float32 — graded evidence in [0, 1]
    boundaries_s: list[float]
    channels: Mapping[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> BoundaryEvidence:
        return cls(
            times_s=np.zeros(0, dtype=np.float64),
            score=np.zeros(0, dtype=np.float32),
            boundaries_s=[],
        )

    def feature_curves(self) -> dict[str, np.ndarray]:
        """The curves the deciders describe: the sub-curves, or the score when
        the modality has none."""
        return dict(self.channels) if self.channels else {"score": self.score}


def grid(duration_s: float, hz: float = GRID_HZ) -> np.ndarray:
    """
    The sampling instants nΔ, n = 0 … ⌊T/Δ⌋, with Δ = 1/hz and T = duration_s.

    The curve fixes how a hint is spread over time, the grid only where it is
    looked at — its spacing can change without redefining a curve.
    """
    if duration_s < 0:
        return np.zeros(0, dtype=np.float64)
    return np.arange(int(np.floor(duration_s * hz + 1e-9)) + 1, dtype=np.float64) / hz


def draw_gaussian(
    score: np.ndarray, times_s: np.ndarray, t_i: float, p_i: float, sigma_s: float
) -> None:
    """
    Merge g_i(t) = p_i · exp(−(t − t_i)² / (2σ²)) into `score` by pointwise
    maximum: the peak lies at t_i with height p_i; cut off beyond 3σ.

    Maximum rather than sum, so that a modality reporting often does not pose
    as certain, and E(t) ≤ max p_i ≤ 1 holds by construction.
    """
    if p_i <= 0.0:
        return
    near = np.abs(times_s - t_i) <= _GAUSS_CUTOFF_SIGMA * sigma_s
    g = p_i * np.exp(-((times_s[near] - t_i) ** 2) / (2.0 * sigma_s ** 2))
    score[near] = np.maximum(score[near], g.astype(np.float32))


def draw_window(
    score: np.ndarray, times_s: np.ndarray, left_s: float, peak_s: float, right_s: float,
    p: float,
) -> None:
    """
    Merge a raised-cosine window into `score` by pointwise maximum, with
    L = left_s, c = peak_s, R = right_s:

        w(t) = p · ½ (1 − cos(π (t − L) / (c − L)))   for L ≤ t ≤ c
        w(t) = p · ½ (1 + cos(π (t − c) / (R − c)))   for c < t ≤ R
        w(t) = 0                                       otherwise

    w(L) = 0, w(c) = p, w(R) = 0. For c = L the rising flank is absent: the
    curve starts at p and falls. Nothing is drawn before L.
    """
    if p <= 0.0:
        return
    w = np.zeros(len(times_s), dtype=np.float64)
    if peak_s > left_s:
        rise = (times_s >= left_s) & (times_s <= peak_s)
        w[rise] = p * 0.5 * (1.0 - np.cos(np.pi * (times_s[rise] - left_s) / (peak_s - left_s)))
    elif peak_s == left_s:
        w[times_s == peak_s] = p
    fall = (times_s > peak_s) & (times_s <= right_s) & (times_s >= left_s)
    w[fall] = p * 0.5 * (1.0 + np.cos(np.pi * (times_s[fall] - peak_s) / (right_s - peak_s)))
    np.maximum(score, w.astype(np.float32), out=score)
