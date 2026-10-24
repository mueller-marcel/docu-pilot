"""
Whether a measured difference is real or an accident of the sessions that
happened to be recorded (C6).

Shapley values and saturation steps are linear combinations of the v(S). They
can therefore be computed PER SESSION and treated like any other measured
quantity: mean, standard deviation and the 95 % t interval over the sessions.

TF1 and TF2 are reported as value and interval. Only the saturation steps (TF3)
are compared with 0 and with ±SESOI (MA 3.3.3, Abb. 3.5).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.stats import t

# Session name -> that session's value of one quantity.
PerSession = Mapping[str, float]
PerSessionBySubset = Mapping[frozenset[str], PerSession]

# Confidence level of every interval.
LEVEL = 0.95

# The five cases of MA 3.3.3, Abb. 3.5.
RELEVANT = "R"         # CI fully outside ±SESOI: statistically and practically relevant
SIGNIFICANT = "S"      # CI excludes 0, partly inside ±SESOI: only statistically relevant
SIGNIFICANT_EQUIVALENT = "SÄ"  # CI excludes 0, fully inside ±SESOI: secured, but practically negligible
EQUIVALENT = "Ä"       # CI contains 0, fully inside ±SESOI: negligible
UNDECIDED = "U"        # CI contains 0 and reaches beyond ±SESOI: no statement


# ── Intervals ─────────────────────────────────────────────────────────────────

def t_interval(values: Sequence[float], level: float = LEVEL) -> tuple[float, float, float]:
    """(estimate, lower, upper) of the mean of per-session values."""
    x = np.asarray(values, dtype=np.float64)
    n = len(x)
    est = float(x.mean()) if n else math.nan
    if n < 2:
        return est, est, est
    half = float(t.ppf(1.0 - (1.0 - level) / 2.0, n - 1)) * float(x.std(ddof=1)) / math.sqrt(n)
    return est, est - half, est + half


def sesoi_from_reference(reference: Mapping[str, Sequence[float]]) -> tuple[float, str]:
    """
    The SESOI (MA 3.3.3) and its provenance: the mean over the sessions of
    1/n_s, n_s the boundaries of session s in the reference annotation.

    The smallest meaningful unit of a segmentation is one boundary — found or
    not (Lakens et al. 2018: SESOI from the smallest unit of the measure, in
    raw units). One more boundary found raises a session's recall by exactly
    1/n_s and its F1 by 2/(n_s + n̂_s), which is 1/n_s when the prediction has
    as many boundaries as the reference. It depends on the reference only,
    never on which coalition scores best, so it is fixed before the run.

    :param reference: session -> boundaries of the primary annotation.
    :raises ValueError: when no session has a boundary.
    """
    counts = [len(b) for b in reference.values() if len(b)]
    if not counts:
        raise ValueError("Keine Session mit Grenzen in der Referenzannotation — keine SESOI.")
    value = float(np.mean([1.0 / n for n in counts]))
    return value, (
        f"Referenzannotation: Mittel von 1/n_s über {len(counts)} Sessions "
        f"(n_s = {min(counts)} … {max(counts)}, Mittel {np.mean(counts):.1f} Grenzen) — "
        f"eine weitere gefundene Grenze"
    )


def classify(lo: float, hi: float, sesoi: float) -> str:
    """
    The five cases of MA 3.3.3 (Abb. 3.5, after Lakens et al. 2018):

        R   CI fully outside [−SESOI, +SESOI]
        SÄ  CI excludes 0 and lies fully inside ±SESOI
        Ä   CI contains 0 and lies fully inside ±SESOI
        S   CI excludes 0 and lies partly inside ±SESOI
        U   CI contains 0 and reaches beyond ±SESOI
    """
    excludes_zero = lo > 0.0 or hi < 0.0
    if lo > sesoi or hi < -sesoi:
        return RELEVANT
    if -sesoi <= lo and hi <= sesoi:
        return SIGNIFICANT_EQUIVALENT if excludes_zero else EQUIVALENT
    if excludes_zero:
        return SIGNIFICANT
    return UNDECIDED


# ── One quantity ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Estimate:
    """A per-session quantity: its mean and 95 % interval over the sessions."""

    name: str
    values: dict[str, float]
    est: float
    lo: float
    hi: float

    def cls(self, sesoi: float) -> str:
        """The class against ±SESOI — used for the saturation steps only."""
        return classify(self.lo, self.hi, sesoi)


def estimate(name: str, values: Mapping[str, float]) -> Estimate:
    """Mean and 95 % t interval of one quantity over the sessions."""
    ordered = dict(sorted(values.items()))
    est, lo, hi = t_interval(list(ordered.values()))
    return Estimate(name, ordered, est, lo, hi)


# ── Per-session game values ───────────────────────────────────────────────────

def _sessions(per_session_by_subset: PerSessionBySubset) -> list[str]:
    names = sorted(next(iter(per_session_by_subset.values())))
    for scores in per_session_by_subset.values():
        if sorted(scores) != names:
            raise KeyError("Alle Teilmengen müssen auf denselben Sessions bewertet sein")
    return names


def _values_of(per_session_by_subset: PerSessionBySubset, session: str) -> dict[frozenset[str], float]:
    return {subset: scores[session] for subset, scores in per_session_by_subset.items()}


def shapley_per_session(
    per_session_by_subset: PerSessionBySubset, players: Sequence[str]
) -> dict[str, dict[str, float]]:
    """φ_i per session: player -> session -> value."""
    from docupilot.evaluation.analysis import shapley

    out: dict[str, dict[str, float]] = {p: {} for p in players}
    for s in _sessions(per_session_by_subset):
        phi = shapley(_values_of(per_session_by_subset, s), players)
        for p in players:
            out[p][s] = phi[p]
    return out


def steps_per_session(
    per_session_by_subset: PerSessionBySubset, players: Sequence[str]
) -> dict[int, dict[str, float]]:
    """Δ(k) per session: mean F1 over coalitions of size k minus size k−1."""
    names = _sessions(per_session_by_subset)

    def size_mean(size: int, s: str) -> float:
        return float(np.mean([
            per_session_by_subset[frozenset(c)][s] for c in combinations(players, size)
        ]))

    return {
        k: {s: size_mean(k, s) - size_mean(k - 1, s) for s in names}
        for k in range(1, len(players) + 1)
    }


def efficiency_errors(
    per_session_by_subset: PerSessionBySubset, players: Sequence[str]
) -> dict[str, float]:
    """|Σ_i φ_i^s − (F1_s(N) − F1_s(∅))| per session — an identity, so anything
    above rounding noise is an implementation error."""
    phi = shapley_per_session(per_session_by_subset, players)
    full, empty = frozenset(players), frozenset()
    return {
        s: abs(sum(phi[p][s] for p in players)
               - (per_session_by_subset[full][s] - per_session_by_subset[empty][s]))
        for s in _sessions(per_session_by_subset)
    }


# ── Saturation point ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SaturationPoint:
    """k*: the smallest number of modalities from which no further one adds a
    relevant gain."""

    k_star: int
    relevant_steps: tuple[int, ...]
    """The steps Δ(k) that are a relevant gain: mean ≥ SESOI and CI above 0."""


def relevant_gain(step: Estimate, sesoi: float) -> bool:
    """A gain counts (MA 2.5) when it is on average at least the SESOI — one
    more boundary — and statistically secured: its interval lies above 0."""
    return step.est >= sesoi and step.lo > 0.0


def saturation_point(steps: Mapping[int, Estimate], sesoi: float) -> SaturationPoint:
    """
    k* (MA 2.5) = the smallest number of modalities from which the gain of
    every further modality is on average below the SESOI or not statistically
    secured: the last relevant step, or 0 when none is. k* equal to the number
    of modalities means no saturation within the measured range.
    """
    relevant = tuple(k for k in sorted(steps) if relevant_gain(steps[k], sesoi))
    return SaturationPoint(max(relevant, default=0), relevant)
