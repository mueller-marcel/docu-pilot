"""
How consistently the one annotator sets boundaries — measured by a blind,
time-shifted repetition of the annotation (test-retest, C5).

The repetition yields the two numbers the evaluation otherwise has to set:

    τ       the accuracy a boundary can be set with at all: the mean deviation
            of the boundaries matched between run 1 and run 2 (formula 3.1)
    SESOI   a difference smaller than what changing the reference from run 1
            to run 2 does to the full model's F1 is annotation noise, not an
            effect (from `sesoi_from_retest`)

Both are measured on every run, never set: τ before it (`measure_tau`), the
SESOI after the primary run (`measure_sesoi`); which SESOI candidate is used is
a recorded decision (Settings.sesoi_rule).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from docupilot.evaluation import metrics

# Tolerances of the consistency F1 — the protocol of the GEBD benchmark.
CONSISTENCY_TAUS_S: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)

# Bins of the distance histogram in the report.
HISTOGRAM_EDGES_S: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, np.inf)


def max_match_distance(g1: Sequence[float]) -> float:
    """
    δ_max = ½ · min_k (a_{k+1} − a_k) over the boundaries of run 1: two marks
    closer than this can never be assigned to two different boundaries.
    0 when run 1 has fewer than two boundaries — no gap, so no pair is formed.
    """
    a = np.sort(np.asarray(g1, dtype=np.float64))
    return 0.5 * float(np.min(np.diff(a))) if len(a) >= 2 else 0.0


def match_runs(
    g1: Sequence[float], g2: Sequence[float]
) -> tuple[list[tuple[float, float]], tuple[list[float], list[float]]]:
    """
    Pair the boundaries of two runs one-to-one with |a_k − b_l| ≤ δ_max: as
    many pairs as possible, ties broken by the smallest total distance
    (Kuhn–Munkres, as metrics.match). Marks without a partner do not enter τ.

    :return: the matched pairs (a_k, b_l), and the unmatched boundaries of run 1
        and run 2.
    """
    d_max = max_match_distance(g1)
    pairs = metrics.match(g1, g2, d_max).pairs if d_max > 0 else []
    used1 = {a for a, _ in pairs}
    used2 = {b for _, b in pairs}
    return pairs, ([t for t in g1 if t not in used1], [t for t in g2 if t not in used2])


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
    distances_s: list[float]
    """|Δt_i| of every matched pair, over all retest sessions."""
    unmatched: int
    consistency: float
    sesoi_diffs: dict[str, float] = field(default_factory=dict)
    sesoi: dict[str, float] = field(default_factory=dict)

    @property
    def tau_s(self) -> float:
        """τ = (1/N) Σ |Δt_i| (formula 3.1); NaN without a pair."""
        return float(np.mean(self.distances_s)) if self.distances_s else float("nan")

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
        distances_s=[abs(a - b) for a, b in pairs],
        unmatched=unmatched,
        consistency=float(np.mean([consistency_f1(run1[s], run2[s]) for s in sessions])),
        sesoi_diffs=diffs,
        sesoi=sesoi_candidates(diffs) if diffs else {},
    )


def measure_tau(
    run1: Mapping[str, Sequence[float]], run2: Mapping[str, Sequence[float]]
) -> tuple[float, str]:
    """
    τ (formula 3.1) and its provenance, measured before a run.

    :raises ValueError: when no session was annotated twice, or no boundary of
        the repetition could be paired — then τ cannot be measured and no run
        may start.
    """
    result = analyse(run1, run2)
    if result is None:
        raise ValueError(
            "Keine Wiederholungsannotation (Lauf 2) im Korpus — τ lässt sich nicht messen. "
            "Mindestens eine Session blind ein zweites Mal annotieren."
        )
    if not result.distances_s:
        raise ValueError(
            "Die Wiederholungsannotation liefert keine gepaarten Grenzen — τ lässt sich "
            "nicht messen."
        )
    return result.tau_s, (
        f"Wiederholungsannotation: mittlere Abweichung (Formel 3.1), "
        f"{len(result.distances_s)} Paare aus {len(result.sessions)} Sessions"
    )


def measure_sesoi(
    run1: Mapping[str, Sequence[float]],
    run2: Mapping[str, Sequence[float]],
    predictions: Mapping[str, Sequence[float]],
    tau_s: float,
    rule: str,
) -> tuple[float, str]:
    """
    The SESOI by the recorded rule and its provenance, measured from the full
    model's predictions of the primary run.

    :raises ValueError: when the retest sessions give no positive SESOI.
    """
    sessions = [s for s in run2 if s in run1]
    diffs = sesoi_from_retest({s: predictions[s] for s in sessions if s in predictions},
                              run1, run2, tau_s)
    value = sesoi_candidates(diffs).get(rule, np.nan)
    if not value > 0:
        raise ValueError(
            f"SESOI ({rule}) aus {len(diffs)} Retest-Session(s) = {value} — nicht verwendbar; "
            "die Wiederholungsannotation verändert die F1 des vollen Modells nicht messbar."
        )
    return float(value), (
        f"Wiederholungsannotation: {rule} der F1-Differenzen des vollen Modells, "
        f"{len(diffs)} Sessions"
    )
