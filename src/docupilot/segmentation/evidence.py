"""
What every modality returns, and the two ways of drawing evidence onto a curve.

The shared vocabulary of the segmentation package — deliberately the only thing
the three modality modules have in common.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from docupilot.config import GRID_HZ

__all__ = [
    "BOUNDARY_THRESHOLD", "GRID_HZ", "BoundaryEvidence",
    "apply_gaussian", "apply_window", "grid",
]

# At or above this probability a modality commits to a boundary.
BOUNDARY_THRESHOLD = 0.5


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
    """Timestamps of a `hz`-spaced grid spanning [0, duration_s]."""
    return np.arange(max(0, int(round(duration_s * hz))), dtype=np.float64) / hz


def apply_gaussian(score: np.ndarray, center: int, value: float, spread: int) -> None:
    """Write a symmetric Gaussian peak into `score`; overlapping peaks: max wins."""
    n = len(score)
    if not 0 <= center < n:
        return
    lo, hi = max(0, center - spread), min(n, center + spread + 1)
    offsets = np.arange(lo, hi) - center
    g = np.exp(-0.5 * (offsets / max(spread / 2.0, 1.0)) ** 2)
    score[lo:hi] = np.maximum(score[lo:hi], (value * g).astype(np.float32))


def apply_window(score: np.ndarray, lo: int, hi: int, peak: int, value: float) -> int:
    """
    Write a raised-cosine bump over [lo, hi] peaking at `peak`: zero at both
    edges, `value` at the peak, asymmetric when the peak is off centre.

    :return: the clamped peak index, which the recording's end may have moved.
    """
    n = len(score)
    lo, hi = max(0, lo), min(n - 1, hi)
    if hi <= lo or value <= 0.0:
        return min(max(peak, 0), max(n - 1, 0))
    peak = min(max(peak, lo), hi)

    x = np.arange(lo, hi + 1, dtype=np.float64)
    shape = np.zeros_like(x)

    rise = x <= peak
    if peak > lo:
        shape[rise] = 0.5 * (1.0 - np.cos(np.pi * (x[rise] - lo) / (peak - lo)))
    else:
        shape[rise] = 1.0                      # peak sits on the left edge

    fall = ~rise
    if hi > peak:
        shape[fall] = 0.5 * (1.0 + np.cos(np.pi * (x[fall] - peak) / (hi - peak)))

    score[lo:hi + 1] = np.maximum(score[lo:hi + 1], (value * shape).astype(np.float32))
    return peak
