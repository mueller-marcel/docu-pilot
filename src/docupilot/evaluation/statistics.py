"""
Whether a measured difference is real or an accident of the sessions that
happened to be recorded (C6).

Shapley values, interaction indices and saturation steps are linear
combinations of the v(S). They can therefore be computed PER SESSION and
treated like any other measured quantity: mean, standard deviation, t
interval. Everything here works on those per-session values.

Two intervals are always computed. The naive one treats the session values as
independent. They are not: the training sets of two leave-one-session-out
folds overlap to (n−2)/(n−1), which Nadeau & Bengio (2003) correct by using
1/n + n_test/n_train instead of 1/n in the variance — here 1/n + 1/(n−1). The
correction is derived for random subsamples and is a conservative
approximation for leave-one-session-out. Which interval is primary is a
recorded decision (Settings.ci_method).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.stats import norm, t

from docupilot.config import Settings

# Session name -> that session's value of one quantity.
PerSession = Mapping[str, float]
PerSessionBySubset = Mapping[frozenset[str], PerSession]

METHODS: tuple[str, ...] = ("naive", "nadeau_bengio")

# The classes of the decision rule, in the order they are checked.
EQUIVALENT = "Ä"       # CI fully inside ±SESOI: provably negligible
RELEVANT = "R"         # CI excludes 0 and the estimate reaches the SESOI
SIGNIFICANT = "S"      # CI excludes 0, practical size open
UNDECIDED = "U"        # anything else

CLASS_TEXT = {
    EQUIVALENT: "nachweislich vernachlässigbar (Äquivalenz)",
    RELEVANT: "statistisch und praktisch relevant",
    SIGNIFICANT: "statistisch gesichert, praktische Größe offen",
    UNDECIDED: "unentscheidbar",
}

_POWER = 0.80
_ALPHA_POWER = 0.05
_MAX_N = 100_000


# ── Intervals ─────────────────────────────────────────────────────────────────

def variance_factor(n: int, method: str) -> float:
    """f(n): 1/n (naive) or 1/n + 1/(n−1) (Nadeau–Bengio for leave-one-out)."""
    if n < 2:
        return math.inf
    return 1.0 / n if method == "naive" else 1.0 / n + 1.0 / (n - 1)


def t_interval(values: Sequence[float], level: float, method: str) -> tuple[float, float, float]:
    """
    (estimate, lower, upper) of the mean of per-session values.

    :param level: confidence level, e.g. 0.95.
    :param method: "naive" or "nadeau_bengio".
    """
    x = np.asarray(values, dtype=np.float64)
    n = len(x)
    est = float(x.mean()) if n else math.nan
    if n < 2:
        return est, est, est
    sd = float(x.std(ddof=1))
    half = float(t.ppf(1.0 - (1.0 - level) / 2.0, n - 1)) * sd * math.sqrt(variance_factor(n, method))
    return est, est - half, est + half


def classify(est: float, lo: float, hi: float, sesoi: float) -> str:
    """
    The decision rule, checked in this order:

        Ä  CI fully inside [−SESOI, +SESOI]
        R  CI excludes 0 and |estimate| ≥ SESOI
        S  CI excludes 0 otherwise
        U  anything else

    The spec states R and S for a positive effect (lower bound > 0); the
    mirror image is applied to a negative one, so a CI that is entirely
    negative — a redundancy, a modality that hurts — is not called undecided.
    """
    if -sesoi <= lo and hi <= sesoi:
        return EQUIVALENT
    if lo > 0.0 or hi < 0.0:
        return RELEVANT if abs(est) >= sesoi else SIGNIFICANT
    return UNDECIDED


# ── Sample size per contrast ──────────────────────────────────────────────────

def minimum_detectable_effect(sd: float, n: int, method: str) -> float:
    """MDE = (z_0.975 + z_0.8) · sd · √f(n) at α = 0.05 and power 0.8."""
    if n < 2:
        return math.inf
    z = norm.ppf(1.0 - _ALPHA_POWER / 2.0) + norm.ppf(_POWER)
    return float(z * sd * math.sqrt(variance_factor(n, method)))


def required_sessions(sd: float, sesoi: float, method: str) -> int:
    """n_req: the smallest n whose MDE is at most the SESOI."""
    if sd <= 0:
        return 2
    for n in range(2, _MAX_N):
        if minimum_detectable_effect(sd, n, method) <= sesoi:
            return n
    return _MAX_N


def equivalence_sessions(sd: float, sesoi: float, method: str, level: float) -> int:
    """n_eq: the smallest n with t_{n−1} · sd · √f(n) ≤ SESOI — how many
    sessions a zero effect needs to be shown negligible."""
    if sd <= 0:
        return 2
    q = 1.0 - (1.0 - level) / 2.0
    for n in range(2, _MAX_N):
        if float(t.ppf(q, n - 1)) * sd * math.sqrt(variance_factor(n, method)) <= sesoi:
            return n
    return _MAX_N


# ── One quantity ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Verdict:
    """One quantity under one interval method."""

    lo: float
    hi: float
    cls: str
    mde: float
    n_req: int
    n_eq: int


@dataclass(frozen=True)
class Estimate:
    """A per-session quantity: its mean, spread, and the verdict per method."""

    name: str
    values: dict[str, float]
    est: float
    sd: float
    n: int
    level: float
    verdicts: dict[str, Verdict]

    def verdict(self, method: str) -> Verdict:
        return self.verdicts[method]


def estimate(name: str, values: Mapping[str, float], sesoi: float, level: float) -> Estimate:
    """Mean, sd, both intervals, classes and sample sizes of one quantity."""
    ordered = dict(sorted(values.items()))
    x = list(ordered.values())
    n = len(x)
    sd = float(np.std(x, ddof=1)) if n >= 2 else 0.0
    verdicts = {}
    for method in METHODS:
        est, lo, hi = t_interval(x, level, method)
        verdicts[method] = Verdict(
            lo=lo, hi=hi, cls=classify(est, lo, hi, sesoi),
            mde=minimum_detectable_effect(sd, n, method),
            n_req=required_sessions(sd, sesoi, method),
            n_eq=equivalence_sessions(sd, sesoi, method, level),
        )
    return Estimate(name, ordered, float(np.mean(x)) if n else math.nan, sd, n, level, verdicts)


def family_level(settings: Settings, family_size: int) -> float:
    """The confidence level of one member of a family of comparisons."""
    if settings.multiplicity == "bonferroni" and family_size > 1:
        return 1.0 - (1.0 - settings.ci_level) / family_size
    return settings.ci_level


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


def interaction_per_session(
    per_session_by_subset: PerSessionBySubset, players: Sequence[str]
) -> dict[tuple[str, str], dict[str, float]]:
    """I_ij per session: pair -> session -> value."""
    from docupilot.evaluation.analysis import interaction

    out: dict[tuple[str, str], dict[str, float]] = {pair: {} for pair in combinations(players, 2)}
    for s in _sessions(per_session_by_subset):
        index = interaction(_values_of(per_session_by_subset, s), players)
        for pair in out:
            out[pair][s] = index[pair]
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


def linearity_error(
    per_session_by_subset: PerSessionBySubset, players: Sequence[str]
) -> float:
    """
    How far the mean of the per-session Shapley values misses the Shapley
    values of the averaged v(S). Linearity makes them equal; the check guards
    the per-session route against a subset scored on different sessions.
    """
    from docupilot.evaluation.analysis import shapley

    names = _sessions(per_session_by_subset)
    averaged = {k: float(np.mean([v[s] for s in names])) for k, v in per_session_by_subset.items()}
    phi_mean = shapley(averaged, players)
    phi_s = shapley_per_session(per_session_by_subset, players)
    return max(abs(float(np.mean(list(phi_s[p].values()))) - phi_mean[p]) for p in players)


# ── Saturation point ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SaturationPoint:
    """k*: the smallest k from which every later step is negligible."""

    k_star: int | None
    """None when a later step is undecided — no saturation statement then."""
    undecided_steps: tuple[int, ...]
    n_for_equivalence: int
    """Sessions an equivalence proof of the undecided steps would need."""

    @property
    def decidable(self) -> bool:
        return self.k_star is not None


def saturation_point(steps: Mapping[int, Estimate], method: str) -> SaturationPoint:
    """
    k* = smallest k such that every Δ(j), j > k, is in class Ä. When the last
    step outside Ä is undecided, the answer is "not decidable", with the
    sessions an equivalence proof would need.
    """
    ks = sorted(steps)
    not_eq = [k for k in ks if steps[k].verdict(method).cls != EQUIVALENT]
    if not not_eq:
        return SaturationPoint(0, (), 0)
    last = max(not_eq)
    if steps[last].verdict(method).cls != UNDECIDED:
        return SaturationPoint(last, (), 0)
    undecided = tuple(k for k in ks if k >= last and steps[k].verdict(method).cls == UNDECIDED)
    return SaturationPoint(
        None, undecided, max(steps[k].verdict(method).n_eq for k in undecided)
    )


# ── Control: cluster bootstrap over sessions ──────────────────────────────────

def cluster_bootstrap(
    values: Mapping[str, float], level: float, n_draws: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """
    Percentile interval of the mean by resampling SESSIONS. A control only,
    reported in the appendix: for a linear quantity resampling the per-session
    values is the same as recomputing every coalition inside each resample.
    """
    x = np.asarray(list(values.values()), dtype=np.float64)
    if len(x) < 2:
        m = float(x.mean()) if len(x) else math.nan
        return m, m
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_draws, len(x)))].mean(axis=1)
    alpha = 1.0 - level
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
