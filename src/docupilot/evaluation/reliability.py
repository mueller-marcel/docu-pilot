"""
How consistently the one annotator sets boundaries — measured by a blind,
time-shifted repetition of the annotation (test-retest, C5).

The repetition yields the two numbers the evaluation otherwise has to set:

    τ       the tolerance may not be smaller than the annotator's own
            repetition deviation (from the distances of matched boundaries)
    SESOI   a difference smaller than what changing the reference from run 1
            to run 2 does to the full model's F1 is annotation noise, not an
            effect (from `sesoi_from_retest`)

Every candidate statistic is computed. Which one defines τ and the SESOI is a
recorded decision (Settings.tau_rule, Settings.sesoi_rule); `derive_settings`
applies it and writes the provenance next to the value.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from docupilot.config import Settings
from docupilot.evaluation import metrics

# Two boundaries further apart than this are not the same boundary set twice.
MATCH_MAX_DIST_S = 3.0

# Tolerances of the consistency F1 — the protocol of the GEBD benchmark.
CONSISTENCY_TAUS_S: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)

# Bins of the distance histogram in the report.
HISTOGRAM_EDGES_S: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, MATCH_MAX_DIST_S)


def match_runs(
    g1: Sequence[float], g2: Sequence[float], max_dist: float = MATCH_MAX_DIST_S
) -> tuple[list[tuple[float, float]], tuple[list[float], list[float]]]:
    """
    Pair the boundaries of two runs one-to-one (Kuhn–Munkres, as metrics.match).

    :return: the matched pairs (t1, t2), and the unmatched boundaries of run 1
        and run 2.
    """
    pairs = metrics.match(g1, g2, max_dist).pairs
    used1 = {a for a, _ in pairs}
    used2 = {b for _, b in pairs}
    return pairs, ([t for t in g1 if t not in used1], [t for t in g2 if t not in used2])


def tau_stats(pairs: Sequence[tuple[float, float]]) -> dict[str, float]:
    """Mean, median, P90 and P95 of |t1 − t2| over all matched pairs, and n."""
    d = np.abs(np.asarray([a - b for a, b in pairs], dtype=np.float64))
    if len(d) == 0:
        return {"mean": np.nan, "median": np.nan, "p90": np.nan, "p95": np.nan, "n": 0}
    return {
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "p90": float(np.percentile(d, 90)),
        "p95": float(np.percentile(d, 95)),
        "n": int(len(d)),
    }


def consistency_f1(
    g1: Sequence[float], g2: Sequence[float], taus: Sequence[float] = CONSISTENCY_TAUS_S
) -> float:
    """Mean F1 of run 2 against run 1 over the tolerances."""
    return float(np.mean([metrics.match(g1, g2, tau).f1 for tau in taus]))


def sesoi_from_retest(
    predictions: Mapping[str, Sequence[float]],
    run1: Mapping[str, Sequence[float]],
    run2: Mapping[str, Sequence[float]],
    tau_s: float,
) -> dict[str, float]:
    """
    Per retest session, F1(pred, run 1) − F1(pred, run 2) of the full model's
    predictions: how much the score moves when only the annotation is repeated.

    :return: session -> difference, for the sessions present in all three.
    """
    return {
        s: metrics.match(run1[s], pred, tau_s).f1 - metrics.match(run2[s], pred, tau_s).f1
        for s, pred in predictions.items()
        if s in run1 and s in run2
    }


def sesoi_candidates(diffs: Mapping[str, float]) -> dict[str, float]:
    """The two SESOI candidates: mean(|diff|) and sd(diff)."""
    x = np.asarray(list(diffs.values()), dtype=np.float64)
    return {
        "mean_abs": float(np.abs(x).mean()) if len(x) else np.nan,
        "sd": float(x.std(ddof=1)) if len(x) >= 2 else np.nan,
    }


@dataclass(frozen=True)
class Reliability:
    """The test-retest analysis of one corpus."""

    sessions: tuple[str, ...]
    tau: dict[str, float]
    distances_s: list[float]
    unmatched: int
    consistency: float
    sesoi_diffs: dict[str, float] = field(default_factory=dict)
    sesoi: dict[str, float] = field(default_factory=dict)

    def histogram(self) -> list[tuple[float, float, int]]:
        counts, edges = np.histogram(self.distances_s, bins=HISTOGRAM_EDGES_S)
        return [(float(edges[i]), float(edges[i + 1]), int(c)) for i, c in enumerate(counts)]


def analyse(
    run1: Mapping[str, Sequence[float]],
    run2: Mapping[str, Sequence[float]],
    predictions: Mapping[str, Sequence[float]] | None = None,
    tau_s: float = 1.0,
) -> Reliability | None:
    """
    The whole reliability analysis over the sessions annotated twice.

    :param run1: session -> boundaries of the annotation.
    :param run2: session -> boundaries of the blind repetition.
    :param predictions: session -> the full model's predictions, for the SESOI;
        omitted before the experiment has run.
    :return: None when no session has a repetition.
    """
    sessions = tuple(sorted(s for s in run2 if s in run1))
    if not sessions:
        return None
    pairs, unmatched = [], 0
    for s in sessions:
        p, (u1, u2) = match_runs(run1[s], run2[s])
        pairs.extend(p)
        unmatched += len(u1) + len(u2)
    diffs = sesoi_from_retest(
        {s: predictions[s] for s in sessions if s in predictions}, run1, run2, tau_s
    ) if predictions else {}
    return Reliability(
        sessions=sessions,
        tau=tau_stats(pairs),
        distances_s=[abs(a - b) for a, b in pairs],
        unmatched=unmatched,
        consistency=float(np.mean([consistency_f1(run1[s], run2[s]) for s in sessions])),
        sesoi_diffs=diffs,
        sesoi=sesoi_candidates(diffs) if diffs else {},
    )


def derive_settings(settings: Settings, reliability: Reliability) -> Settings:
    """
    The settings with τ and the SESOI taken from the retest, by the recorded
    rules, each with its provenance. A quantity the retest cannot provide yet
    (no pairs; no predictions) keeps its current value and source.
    """
    changes: dict[str, object] = {}
    n = len(reliability.sessions)
    tau = reliability.tau.get(settings.tau_rule, np.nan)
    if tau == tau and tau > 0:
        changes["tau_primary_s"] = round(float(tau), 3)
        changes["tau_source"] = (
            f"Wiederholungsannotation: {settings.tau_rule} der Abstände, "
            f"n = {reliability.tau['n']} Paare aus {n} Sessions"
        )
    sesoi = reliability.sesoi.get(settings.sesoi_rule, np.nan)
    if sesoi == sesoi and sesoi > 0:
        changes["sesoi"] = round(float(sesoi), 4)
        changes["sesoi_source"] = (
            f"Wiederholungsannotation: {settings.sesoi_rule} der F1-Differenzen des "
            f"vollen Modells, {len(reliability.sesoi_diffs)} Sessions"
        )
    return settings.replace(**changes) if changes else settings
