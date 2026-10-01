"""
The finished evaluation of one corpus, and the sentences that state it.

Everything the window and the PDF show is computed here, without Qt, so the
same numbers can be produced by a script — the figures for the thesis must be
reproducible without opening a window.

The report answers the three sub-questions on the primary run and surrounds
the answer with what a reviewer will ask for: both confidence intervals and the
class of every quantity (C6), the sample size each contrast would need, the
method comparison (C12), the recall ceiling, the
in-fold calibration (C8), the annotation's reliability (C5)
and where every model verdict came from (C11).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np

from docupilot.config import PARAMETERS, Settings
from docupilot.evaluation import (
    analysis, calibration, experiment, fusion, metrics, reliability, statistics,
)
from docupilot.evaluation.experiment import ExperimentResult, SessionData
from docupilot.evaluation.statistics import Estimate, SaturationPoint
from docupilot.segmentation import MODALITIES, BoundaryEvidence

# Same colours the feature lanes use, so a modality looks the same everywhere.
COLORS = {"events": "#38bdf8", "audio": "#a78bfa", "video": "#fb923c"}

METHOD_TEXT = {"naive": "naiv", "nadeau_bengio": "korrigiert (Nadeau–Bengio)"}


@dataclass(frozen=True)
class VariantSummary:
    """One run of the method comparison: only ranking and saturation (C12)."""

    label: str
    overrides: dict
    values: dict[frozenset[str], float]
    shapley: dict[str, float]
    saturation: SaturationPoint

    @property
    def ranking(self) -> tuple[str, ...]:
        return tuple(sorted(self.shapley, key=lambda m: -self.shapley[m]))


@dataclass
class Report:
    """Everything the charts and texts need, computed once off the UI thread."""

    settings: Settings
    players: tuple[str, ...]
    n_sessions: int
    rows: list[dict] = field(default_factory=list)
    variant_rows: list[dict] = field(default_factory=list)

    values: dict[frozenset[str], float] = field(default_factory=dict)
    subsets: dict[frozenset[str], Estimate] = field(default_factory=dict)
    shapley: dict[str, Estimate] = field(default_factory=dict)
    interactions: dict[tuple[str, str], Estimate] = field(default_factory=dict)
    steps: dict[int, Estimate] = field(default_factory=dict)
    curve: dict[int, float] = field(default_factory=dict)
    saturation: dict[str, SaturationPoint] = field(default_factory=dict)
    recall_all: Estimate | None = None
    precision_all: Estimate | None = None
    chance: float = 0.0
    efficiency_error: float = 0.0
    linearity_error: float = 0.0

    thresholds: dict[frozenset[str], tuple[float, float, float, float]] = field(default_factory=dict)
    """(mean, min, max, OOB mean) of the chosen threshold per coalition over the folds."""
    folds: list[dict] = field(default_factory=list)
    audio_diagnosis: calibration.AudioCalibration | None = None

    shapley_by_tau: dict[float, dict[str, float]] = field(default_factory=dict)
    f1_all_by_tau: dict[float, float] = field(default_factory=dict)
    shapley_recall: dict[str, float] = field(default_factory=dict)

    variants: list[VariantSummary] = field(default_factory=list)
    timing_credit: dict[str, float] = field(default_factory=dict)
    """v_union({m}) − v_grid({m}): what a modality gains from the others' timing."""
    event_credit: dict[frozenset[str], float] = field(default_factory=dict)
    """v(with C14) − v(rest only) per coalition with events."""

    ceiling: float = float("nan")
    """Recall ceiling: share of boundaries some modality proposes within ±τ."""
    reliability: reliability.Reliability | None = None
    provenance: dict[str, object] = field(default_factory=dict)
    bootstrap: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def method(self) -> str:
        return self.settings.ci_method

    @property
    def tau_s(self) -> float:
        return self.settings.tau_primary_s

    @property
    def relevance_threshold(self) -> float:
        return self.settings.sesoi

    @property
    def total(self) -> float:
        return self.values.get(frozenset(self.players), 0.0)

    @property
    def v_empty(self) -> float:
        return self.values.get(frozenset(), 0.0)

    @property
    def gains(self) -> dict[int, float]:
        return analysis.marginal_gain(self.curve)

    @property
    def f1_all(self) -> Estimate | None:
        return self.subsets.get(frozenset(self.players))

    @property
    def decision_loss(self) -> float:
        """Recall the decider leaves on the table: boundaries some modality
        proposed within τ that the full model still did not accept."""
        if self.recall_all is None:
            return float("nan")
        return self.ceiling - self.recall_all.est


# ── Computation ──────────────────────────────────────────────────────────────

def _estimates(per_session: Mapping, prefix: Callable, settings: Settings, family: int) -> dict:
    level = statistics.family_level(settings, family)
    return {
        key: statistics.estimate(prefix(key), values, settings.sesoi, level)
        for key, values in per_session.items()
    }


def _variant(result: ExperimentResult, overrides: dict, players: Sequence[str]) -> VariantSummary:
    settings = result.settings
    tau = settings.tau_primary_s
    paired = experiment.paired_metric(result.rows, tau)
    steps = _estimates(statistics.steps_per_session(paired, players),
                       lambda k: f"Δ({k})", settings, len(players))
    values = experiment.subset_values(result.rows, tau)
    return VariantSummary(
        label=result.label,
        overrides=dict(overrides),
        values=values,
        shapley=analysis.shapley(values, players),
        saturation=statistics.saturation_point(steps, settings.ci_method),
    )


def _thresholds(rows: Sequence[dict], tau_s: float) -> dict[frozenset[str], tuple[float, float, float, float]]:
    chosen: dict[frozenset[str], list[float]] = {}
    oob: dict[frozenset[str], list[float]] = {}
    for row in rows:
        if row["tau_s"] != tau_s:
            continue
        key = experiment.subset_from_label(row["subset"])
        t = float(row["threshold"])
        if t == t:
            chosen.setdefault(key, []).append(t)
        t_oob = float(row.get("threshold_oob", float("nan")))
        if t_oob == t_oob:
            oob.setdefault(key, []).append(t_oob)
    return {
        k: (float(np.mean(v)), float(np.min(v)), float(np.max(v)),
            float(np.mean(oob[k])) if oob.get(k) else float("nan"))
        for k, v in chosen.items()
    }


def _provenance(data: Sequence[SessionData]) -> dict[str, object]:
    models: set[str] = set()
    dates: list[str] = []
    limit_hits, unknown = [], {}
    for d in data:
        video_raw, audio_raw, events_raw = d.raws.get("video"), d.raws.get("audio"), d.raws.get("events")
        if video_raw is not None:
            for v in video_raw.verdicts:
                models.add(v.judgement.model or "(unbekannt, Alt-Cache)")
                if v.judgement.retrieved_at:
                    dates.append(v.judgement.retrieved_at)
            if video_raw.limit_hit:
                limit_hits.append(d.name)
        if audio_raw is not None and audio_raw.sentences:
            models.add(audio_raw.model or "(unbekannt, Alt-Cache)")
            if audio_raw.retrieved_at:
                dates.append(audio_raw.retrieved_at)
        if events_raw is not None:
            for key, n in events_raw.unknown_keys.items():
                unknown[key] = unknown.get(key, 0) + n
    return {
        "models": sorted(models),
        "retrieved": (min(dates)[:10], max(dates)[:10]) if dates else None,
        "limit_hits": limit_hits,
        "video_calls": {d.name: getattr(d.raws.get("video"), "calls", 0) for d in data},
        "unknown_keys": dict(sorted(unknown.items(), key=lambda kv: -kv[1])),
    }


def recall_ceiling(
    sessions: Sequence[tuple[Mapping[str, BoundaryEvidence], Sequence[float]]],
    players: Sequence[str],
    tau_s: float,
) -> float:
    """
    Share of annotated boundaries with a peak of ANY modality within ±τ,
    averaged over sessions — a boundary no modality proposes cannot be found
    by any decider.

    :param sessions: (curves as the session's fold drew them, boundaries) each.
    """
    shares = []
    for evidence, gt in sessions:
        if not gt:
            continue
        peaks = fusion.candidate_times(evidence, players)
        shares.append(
            np.mean([bool(peaks.size) and float(np.min(np.abs(peaks - g))) <= tau_s for g in gt])
        )
    return float(np.mean(shares)) if shares else float("nan")


def analyse(
    data: Sequence[SessionData],
    primary: ExperimentResult,
    variants: Sequence[tuple[dict, ExperimentResult]] = (),
    players: Sequence[str] = MODALITIES,
) -> Report:
    """
    Turn the result tables into the report the thesis reads.

    :param data: the corpus the primary run was computed on.
    :param primary: the primary run.
    :param variants: (overrides, result) of each method-comparison run.
    :param players: the modalities, in the order the charts list them.
    """
    settings = primary.settings
    players = tuple(players)
    tau = settings.tau_primary_s
    rows = primary.rows
    data = experiment.usable(data, settings.boundary_definition)
    paired = experiment.paired_metric(rows, tau)
    values = experiment.subset_values(rows, tau)
    full = frozenset(players)

    shapley = _estimates(statistics.shapley_per_session(paired, players),
                         lambda m: f"φ {m}", settings, len(players))
    pairs = list(combinations(players, 2))
    interactions = _estimates(statistics.interaction_per_session(paired, players),
                              lambda p: f"I {p[0]}↔{p[1]}", settings, len(pairs))
    steps = _estimates(statistics.steps_per_session(paired, players),
                       lambda k: f"Δ({k})", settings, len(players))
    subsets = _estimates(paired, experiment.subset_label, settings, 1)

    report = Report(
        settings=settings,
        players=players,
        n_sessions=len(paired.get(full, {})),
        rows=list(rows),
        variant_rows=[r for _, v in variants for r in v.rows],
        values=values,
        subsets=subsets,
        shapley=shapley,
        interactions=interactions,
        steps=steps,
        curve=analysis.saturation(values, players),
        saturation={m: statistics.saturation_point(steps, m) for m in statistics.METHODS},
        recall_all=statistics.estimate(
            "Recall", experiment.paired_metric(rows, tau, "recall")[full], settings.sesoi,
            settings.ci_level),
        precision_all=statistics.estimate(
            "Precision", experiment.paired_metric(rows, tau, "precision")[full], settings.sesoi,
            settings.ci_level),
        chance=float(np.mean([
            metrics.chance_level(d.gt_s(settings.boundary_definition), d.duration_s, tau)
            for d in data
        ])) if data else 0.0,
        efficiency_error=max(statistics.efficiency_errors(paired, players).values()),
        linearity_error=statistics.linearity_error(paired, players),
        thresholds=_thresholds(rows, tau),
        folds=[
            {"session": f.held_out, "rest_full_s": f.calibration.params.rest_full_s,
             "anchor": f.calibration.audio.anchor, "offset_s": f.calibration.audio.offset_s,
             "pairs": f.calibration.audio.n_pairs, "match_rate": f.calibration.audio.match_rate,
             "fallback": f.calibration.audio.fallback}
            for f in primary.folds
        ],
        audio_diagnosis=calibration.calibrate_audio(
            [d.raws["audio"] for d in data],
            [d.gt_s(settings.boundary_definition) for d in data],
        ) if data else None,
        shapley_by_tau={
            t: analysis.shapley(experiment.subset_values(rows, t), players)
            for t in experiment.taus_in(rows)
        },
        f1_all_by_tau={t: experiment.subset_values(rows, t)[full] for t in experiment.taus_in(rows)},
        shapley_recall=analysis.shapley(experiment.subset_values(rows, tau, "recall"), players),
        provenance=_provenance(data),
    )

    report.variants = [_variant(result, overrides, players) for overrides, result in variants]
    for overrides, result in variants:
        v = experiment.subset_values(result.rows, tau)
        if overrides == {"candidate_pool": "union"}:
            report.timing_credit = {m: v[frozenset({m})] - values[frozenset({m})] for m in players}
        if overrides == {"event_channels": "rest"}:
            report.event_credit = {s: values[s] - v[s] for s in values if "events" in s}

    by_name = {d.name: d for d in data}
    report.ceiling = recall_ceiling(
        [(f.evidence, by_name[f.held_out].gt_s(settings.boundary_definition))
         for f in primary.folds if f.held_out in by_name],
        players, tau,
    )

    run2 = {d.name: d.gt_s(settings.boundary_definition, 2)
            for d in data if d.has(settings.boundary_definition, 2)}
    if run2:
        run1 = {d.name: d.gt_s(settings.boundary_definition) for d in data}
        report.reliability = reliability.analyse(
            run1, run2, primary.predictions(experiment.subset_label(players)), tau
        )

    if settings.bootstrap_check:
        level = settings.ci_level
        report.bootstrap = {
            e.name: statistics.cluster_bootstrap(e.values, level, seed=settings.seed)
            for e in [*shapley.values(), *interactions.values(), *steps.values()]
        }
    return report


# ── Rows for the charts ──────────────────────────────────────────────────────

def shapley_rows(report: Report) -> list[tuple[str, float, float, float, str]]:
    """(modality, value, ci_low, ci_high, colour), strongest first."""
    rows = []
    for m, e in report.shapley.items():
        v = e.verdict(report.method)
        rows.append((m, e.est, v.lo, v.hi, COLORS.get(m, "#333")))
    return sorted(rows, key=lambda row: -row[1])


def subset_rows(report: Report) -> list[tuple[frozenset[str], float, float, float]]:
    """(subset, F1, ci_low, ci_high), best first."""
    rows = []
    for subset, e in report.subsets.items():
        v = e.verdict(report.method)
        rows.append((subset, e.est, v.lo, v.hi))
    return sorted(rows, key=lambda row: -row[1])


# ── Texts ────────────────────────────────────────────────────────────────────

def _label(subset: frozenset[str]) -> str:
    return experiment.subset_label(subset)


def _ci(e: Estimate | None, method: str, signed: bool = True) -> str:
    if e is None:
        return "–"
    v = e.verdict(method)
    f = "+.3f" if signed else ".3f"
    return f"{e.est:{f}} [{v.lo:{f}}, {v.hi:{f}}]"


def _both(e: Estimate) -> str:
    """Estimate with both intervals and both classes."""
    n, c = e.verdict("naive"), e.verdict("nadeau_bengio")
    return (f"{e.est:+.3f}   naiv [{n.lo:+.3f}, {n.hi:+.3f}] {n.cls}   "
            f"korr. [{c.lo:+.3f}, {c.hi:+.3f}] {c.cls}")


def _saturation_sentence(point: SaturationPoint, n_players: int) -> str:
    if point.k_star is None:
        steps = ", ".join(f"+{k}" for k in point.undecided_steps)
        return (f"nicht entscheidbar — Schritt {steps} unentscheidbar; Äquivalenznachweis "
                f"bräuchte ≈ {point.n_for_equivalence} Sessions")
    if point.k_star >= n_players:
        return f"k* = {point.k_star}: keine Sättigung im gemessenen Bereich"
    return f"k* = {point.k_star}: jeder weitere Schritt nachweislich < SESOI"


def adequacy_text(report: Report) -> str:
    """Headline metrics of the full modality set, and whether the corpus size
    suffices — judged per contrast by n_req, not by one global MDE."""
    method = report.method
    contrasts = [*report.shapley.values(), *report.steps.values()]
    need = max((e.verdict(method).n_req for e in contrasts), default=0)
    need_eq = max((e.verdict(method).n_eq for e in contrasts), default=0)
    lines = [
        f"   Alle Modalitäten:   F1 = {_ci(report.f1_all, method, False)}   ·   "
        f"Recall = {_ci(report.recall_all, method, False)}   ·   "
        f"Precision = {_ci(report.precision_all, method, False)}",
        f"   v(∅) = {report.v_empty:.3f} (Grenzen im festen Abstand der mittleren Segmentlänge)"
        f"   ·   Zufallsniveau F1 = {report.chance:.3f} (gleichverteilte Grenzen, zweite Referenz)",
        f"   n = {report.n_sessions} Sessions   ·   Intervall: {METHOD_TEXT[method]}, "
        f"{report.settings.ci_level:.0%}   ·   SESOI = {report.settings.sesoi:.3f}",
        f"   größtes n_req über Shapley und Δ(k) = {need}   ·   größtes n_eq = {need_eq}",
        "   → Korpus groß genug, um jede Shapley-Differenz und jeden Sättigungsschritt der "
        "Größe SESOI mit 80 % Power zu zeigen." if report.n_sessions >= need else
        f"   → Korpus zu klein für den schwierigsten Kontrast (n = {report.n_sessions} < "
        f"n_req = {need}) — Klassen U sind dann eine Frage der Stichprobengröße.",
    ]
    return "\n".join(lines)


def design_text(report: Report) -> str:
    """What was computed and with which recorded choices, and what the in-fold
    calibration chose."""
    s = report.settings
    lines = [
        f"   Entscheider: {s.decider}   ·   Kandidaten: {s.candidate_pool}"
        f"{' (5-Hz-Raster, für alle Koalitionen identisch)' if s.candidate_pool == 'grid' else ''}"
        f"   ·   Grenzdefinition: {s.boundary_definition}   ·   Ereignismerkmale: {s.event_channels}",
        f"   τ = {s.tau_primary_s:.3f} s ({s.tau_source})",
        f"   SESOI = {s.sesoi:.4f} ({s.sesoi_source})",
        f"   Mehrfachvergleich: {s.multiplicity}   ·   Audio-Peak: {s.audio_peak_mode}   ·   "
        f"Burst-Pause: {s.burst_pause_s:g} s   ·   ASR: {s.asr_alignment}",
        "   Schwelle je Koalition (innere GroupKFold; Mittel [min–max] über die Folds; "
        "OOB nur zum Vergleich):",
    ]
    for subset, (mean, lo, hi, oob) in sorted(report.thresholds.items(),
                                             key=lambda kv: (len(kv[0]), _label(kv[0]))):
        oob_text = "–" if oob != oob else f"{oob:.2f}"
        lines.append(f"      {_label(subset):22s} {mean:.2f}  [{lo:.2f}–{hi:.2f}]   OOB {oob_text}")
    return "\n".join(lines)


def calibration_text(report: Report) -> str:
    """The parameters each fold calibrated, and how well audio announcements
    can be placed in time at all."""
    if not report.folds:
        return "   (keine Folds)"
    rest = [f["rest_full_s"] for f in report.folds]
    anchors = [f["anchor"] for f in report.folds]
    offsets = [f["offset_s"] for f in report.folds]
    rates = [f["match_rate"] for f in report.folds]
    fallbacks = sum(1 for f in report.folds if f["fallback"])
    values, counts = np.unique(rest, return_counts=True)
    lines = [
        "   REST_FULL_S je Fold: " + ", ".join(f"{v:g} s × {c}" for v, c in zip(values, counts)),
        f"   Audio-Anker: Satzbeginn × {anchors.count('begin')}, Satzende × {anchors.count('end')}"
        f"   ·   d: Median {np.median(offsets):+.2f} s [{min(offsets):+.2f}, {max(offsets):+.2f}]"
        f"   ·   Rückfall (< 10 Paare): {fallbacks} Folds",
        f"   Zuordnungsquote (OPERATION-Sätze mit Referenzgrenze im Suchbereich): "
        f"Mittel {np.mean(rates):.2f} [{min(rates):.2f}–{max(rates):.2f}]",
    ]
    diag = report.audio_diagnosis
    if diag is not None and diag.offsets_s.get(diag.anchor):
        edges = (-np.inf, -2, -1, -0.5, 0, 0.5, 1, 2, 4, np.inf)
        counts, _ = np.histogram(diag.offsets_s[diag.anchor], bins=edges)
        labels = ["<−2", "−2…−1", "−1…−½", "−½…0", "0…½", "½…1", "1…2", "2…4", ">4"]
        lines.append(
            f"   Histogramm g − a ({'Satzbeginn' if diag.anchor == 'begin' else 'Satzende'}, "
            "Diagnose über alle Sessions, nicht zur Vorhersage verwendet):"
        )
        lines.append("      " + "  ".join(f"{lab}: {c}" for lab, c in zip(labels, counts)))
    lines.append(
        "   Lesart: Eine niedrige Zuordnungsquote heißt, viele Ansagen haben keine zeitlich "
        "zuordenbare Handlung — ein kleiner Audio-Beitrag ist dann inhaltlich erklärt."
    )
    return "\n".join(lines)


def shapley_text(report: Report) -> str:
    lines = [f"   {m:8s} {_both(e)}" for m, e in
             sorted(report.shapley.items(), key=lambda kv: -kv[1].est)]
    lines.append(
        f"   Σφ = v(N) − v(∅) je Session: max. Abweichung {report.efficiency_error:.1e}   ·   "
        f"Linearität (Mittel der Session-φ = φ der gemittelten v): {report.linearity_error:.1e}"
    )
    lines.append("   Klassen: Ä = vernachlässigbar, R = relevant, S = gesichert, U = unentscheidbar")
    return "\n".join(lines)


def saturation_text(report: Report) -> str:
    """The steps with both intervals and classes, and k* under both methods."""
    lines = [f"   +{k}. Modalität:  {_both(e)}" for k, e in sorted(report.steps.items())]
    for method in statistics.METHODS:
        lines.append(f"   → {METHOD_TEXT[method]}: "
                     + _saturation_sentence(report.saturation[method], len(report.players)))
    lines.append(
        "   Sättigungskurve = Mittel über alle Koalitionen einer Größe; der Schritt ist "
        "je Session gebildet und dann gemittelt (gepaart)."
    )
    return "\n".join(lines)


def interaction_text(report: Report) -> str:
    return "\n".join(
        f"   {a} ↔ {b}:  {_both(e)}" for (a, b), e in report.interactions.items()
    ) + "\n   Positiv = Synergie, negativ = Redundanz (dieselbe Information)."


def contrasts_text(report: Report) -> str:
    """MDE and required sessions per contrast, from its own paired spread."""
    n = report.n_sessions
    lines = [f"   Kontrast            sd_d    MDE n={n} (naiv/korr.)   n_req (naiv/korr.)   "
             f"n_eq (naiv/korr.)"]
    for e in [*report.shapley.values(), *report.steps.values(), *report.interactions.values()]:
        a, b = e.verdict("naive"), e.verdict("nadeau_bengio")
        lines.append(
            f"   {e.name:18s} {e.sd:6.3f}   {a.mde:6.3f} / {b.mde:6.3f}          "
            f"{a.n_req:5d} / {b.n_req:<5d}        {a.n_eq:5d} / {b.n_eq:<5d}"
        )
    lines.append(
        f"   SESOI = {report.settings.sesoi:.3f}; α = 0,05; Power 0,8. Die Korrektur nach "
        "Nadeau–Bengio verdoppelt die benötigten Sessions ungefähr."
    )
    return "\n".join(lines)


def methods_text(report: Report) -> str:
    """The method comparison: ranking and saturation class only — no further tests."""
    lines = [
        f"   Primär ({report.settings.decider}·{report.settings.candidate_pool}):  Rang "
        + " > ".join(sorted(report.shapley, key=lambda m: -report.shapley[m].est))
        + "   ·   " + _saturation_sentence(report.saturation[report.method], len(report.players))
    ]
    for v in report.variants:
        change = ", ".join(f"{k}={val}" for k, val in v.overrides.items())
        lines.append(
            f"   {change:28s} Rang {' > '.join(v.ranking):26s} F1 alle "
            f"{v.values.get(frozenset(report.players), float('nan')):.3f}   "
            + _saturation_sentence(v.saturation, len(report.players))
        )
    if not report.variants:
        lines.append("   (keine Varianten gerechnet)")
    if report.timing_credit:
        lines.append(
            "   Zeitpunkt-Kredit v_union({m}) − v_grid({m}): "
            + ", ".join(f"{m} {c:+.3f}" for m, c in report.timing_credit.items())
        )
    if report.event_credit:
        lines.append(
            "   Ereignismerkmale (C14) v(mit) − v(nur Ruhe): "
            + ", ".join(f"{_label(s)} {c:+.3f}" for s, c in
                        sorted(report.event_credit.items(), key=lambda kv: len(kv[0])))
            + " — wie stark der Ereignisbeitrag vom Extraktor abhängt."
        )
    return "\n".join(lines)


def robustness_text(report: Report) -> str:
    """Shapley points across the tolerance sweep and by recall."""
    taus = sorted(report.shapley_by_tau)
    if not taus:
        return "   (kein Toleranz-Sweep)"
    players = list(report.players)
    lines = ["   τ [s]      " + "".join(f"{tau:>8.2f}" for tau in taus)]
    for m in players:
        lines.append(f"   {m:10s}" + "".join(f"{report.shapley_by_tau[tau][m]:>8.3f}" for tau in taus))
    lines.append("   F1 alle   " + "".join(f"{report.f1_all_by_tau[tau]:>8.3f}" for tau in taus))
    ranks = {tuple(sorted(players, key=lambda m: -report.shapley_by_tau[tau][m])) for tau in taus}
    lines.append(
        f"   Shapley nach Recall (τ = {report.tau_s:.2f} s): "
        + ", ".join(f"{m} {v:+.3f}" for m, v in report.shapley_recall.items())
    )
    lines.append(
        "   → Rangfolge über alle τ stabil." if len(ranks) == 1
        else f"   → Rangfolge wechselt mit τ ({len(ranks)} verschiedene Ordnungen)."
    )
    return "\n".join(lines)


def ceiling_text(report: Report) -> str:
    """Where the recall is lost: before the decider (no candidate) or in it."""
    if report.ceiling != report.ceiling or report.recall_all is None:
        return "   (keine Obergrenze berechnet)"
    return "\n".join([
        f"   Recall-Obergrenze (irgendeine Modalität schlägt die Grenze in ±τ vor): "
        f"{report.ceiling:.3f}",
        f"   Erreichter Recall (alle Modalitäten): {report.recall_all.est:.3f}   ·   "
        f"Entscheidungsverlust: {report.decision_loss:.3f}",
        f"   Vorschlagsverlust (keine Modalität schlägt vor): {1.0 - report.ceiling:.3f}",
    ])


def reliability_text(report: Report) -> str:
    """The blind repetition of the annotation (C5)."""
    r = report.reliability
    if r is None:
        return ("   (keine Wiederholungsannotation — Lauf 2 im Annotationsfenster anlegen; "
                "τ und SESOI sind bis dahin gesetzt, nicht hergeleitet)")
    t = r.tau
    lines = [
        f"   Retest-Sessions: {len(r.sessions)}   ·   gepaarte Grenzen: {t['n']}   ·   "
        f"ungepaart: {r.unmatched}",
        f"   |t1 − t2|: Mittel {t['mean']:.3f} s · Median {t['median']:.3f} s · "
        f"P90 {t['p90']:.3f} s · P95 {t['p95']:.3f} s",
        "   Abstände: " + "  ".join(f"[{lo:g}, {hi:g}) s: {c}" for lo, hi, c in r.histogram()),
        f"   Konsistenz-F1 (τ = 0,2 … 1,0 s gemittelt): {r.consistency:.3f}",
    ]
    if r.sesoi:
        lines.append(
            f"   SESOI-Kandidaten (F1 Lauf 1 − Lauf 2 des vollen Modells, {len(r.sesoi_diffs)} "
            f"Sessions): mean(|diff|) = {r.sesoi['mean_abs']:.4f} · sd(diff) = {r.sesoi['sd']:.4f}"
        )
    lines.append(
        f"   Regeln: τ = {report.settings.tau_rule}, SESOI = {report.settings.sesoi_rule} "
        "(Entscheidung N5); übernommen erst, wenn sie in die Einstellungen geschrieben sind."
    )
    return "\n".join(lines)


def provenance_text(report: Report) -> str:
    """Which model answered when, the budget cap, and unclassified keys (C11, C14)."""
    p = report.provenance
    retrieved = p.get("retrieved")
    unknown = p.get("unknown_keys") or {}
    hits = p.get("limit_hits") or []
    calls = p.get("video_calls") or {}
    lines = [
        "   Modelle: " + (", ".join(p.get("models") or []) or "–"),
        "   Abrufdatum: " + (f"{retrieved[0]} … {retrieved[1]}" if retrieved else "–"),
        "   VLM-Aufrufe je Session: " + (
            f"Median {int(np.median(list(calls.values())))}, max {max(calls.values())}"
            if calls else "–"
        ),
        "   _MAX_CALLS erreicht in: " + (", ".join(hits) if hits else "keiner Session"),
        f"   Unbekannte Tasten (→ „sonstige“): {sum(unknown.values())}"
        + (" — " + ", ".join(f"{k} × {n}" for k, n in list(unknown.items())[:10]) if unknown else ""),
    ]
    return "\n".join(lines)


def appendix_text(report: Report) -> str:
    """The parameter table (C8) and, when requested, the bootstrap control."""
    lines = ["   Parameter                               Wert          Klasse       Behandlung"]
    for p in PARAMETERS:
        lines.append(f"   {p.name:38s}  {p.value:12s}  {p.kind:11s}  {p.handling}")
    if report.bootstrap:
        lines.append("   Kontrolle: Cluster-Bootstrap über Sessions (Perzentil):")
        for name, (lo, hi) in report.bootstrap.items():
            lines.append(f"      {name:18s} [{lo:+.3f}, {hi:+.3f}]")
    return "\n".join(lines)


def status_line(report: Report) -> str:
    """The one-line summary the window shows when a run completes."""
    f1 = report.f1_all.est if report.f1_all is not None else float("nan")
    return (
        f"Fertig · {report.n_sessions} Sessions · τ = {report.tau_s:.2f} s · "
        f"F1 {f1:.3f} · v(∅) {report.v_empty:.3f} · Zufallsniveau {report.chance:.3f}"
        + (f"  ·  ⚠ Shapley-Effizienzfehler {report.efficiency_error:.1e}"
           if report.efficiency_error > 1e-9 else "")
    )


def header_lines(report: Report, corpus_root: Path | None, created: str) -> list[str]:
    """The metadata block at the top of the PDF report."""
    s = report.settings
    return [
        f"Korpus: {corpus_root if corpus_root else '—'}",
        f"Erstellt: {created}",
        f"Sessions: {report.n_sessions}   ·   Grenzdefinition: {s.boundary_definition}",
        f"Verfahren: {s.decider}, Kandidaten {s.candidate_pool}, Schwelle über innere "
        "GroupKFold, v(∅) = informationslose Baseline, Leave-one-session-out",
        f"τ = {s.tau_primary_s:.2f} s   ·   SESOI = {s.sesoi:.3f}   ·   Intervall "
        f"{METHOD_TEXT[s.ci_method]} {s.ci_level:.0%}",
    ]


@dataclass(frozen=True)
class Section:
    """One block of the report: a heading, a one-line reading aid, and either a
    chart name or a text body (or both, chart first)."""

    title: str
    hint: str
    chart: str | None = None      # "shapley" | "subsets" | "saturation"
    text: str | None = None


def sections(report: Report) -> list[Section]:
    """
    The report in reading order — shared by the window and the PDF so both
    present the same sections with the same headings.
    """
    return [
        Section("Kennzahlen & statistische Aussagekraft",
                "F1 der vollen Modalitätsmenge, die informationslose Baseline v(∅) und ob "
                "der Korpus je Kontrast groß genug ist.",
                text=adequacy_text(report)),
        Section("Versuchsdesign",
                "Die festgelegten Entscheidungen und die Schwelle je Koalition.",
                text=design_text(report)),
        Section("Shapley-Werte · marginaler Beitrag je Modalität (TF2)",
                "Balken = Beitrag, Whisker = primäres Konfidenzintervall. Darunter beide "
                "Intervalle und die Klasse.",
                chart="shapley", text=shapley_text(report)),
        Section("Alle 8 Modalitätskombinationen (TF1)",
                "Gefüllter Punkt = Modalität enthalten. Rote Linie = Zufallsniveau.",
                chart="subsets"),
        Section("Sättigung (TF3)",
                "Mittlerer F1 je Anzahl Modalitäten; Zahlen dazwischen = Zugewinn Δ(k).",
                chart="saturation", text=saturation_text(report)),
        Section("Interaktionsindex · Synergie und Redundanz (TF3)",
                "Paarweise Interaktion je Session, dann gemittelt.",
                text=interaction_text(report)),
        Section("Benötigte Sessions je Kontrast",
                "MDE, n_req und n_eq aus der gepaarten Streuung jedes Kontrasts.",
                text=contrasts_text(report)),
        Section("Methodenvergleich",
                "Varianten per Konfiguration; berichtet werden nur Rangfolge und Sättigung.",
                text=methods_text(report)),
        Section("In-fold kalibrierte Parameter",
                "Was jeder Fold auf seinen Trainingssessions gewählt hat.",
                text=calibration_text(report)),
        Section("Robustheit · Toleranz-Sweep und Recall-Attribution",
                "Shapley-Werte je τ und nach Recall statt F1.",
                text=robustness_text(report)),
        Section("Recall-Obergrenze und Verlustzerlegung",
                "Was kein Entscheider erreichen kann gegen das, was dieser liegen lässt.",
                text=ceiling_text(report)),
        Section("Reliabilität der Annotation",
                "Blinde, zeitversetzte Wiederholung: Abstände, Konsistenz, SESOI-Kandidaten.",
                text=reliability_text(report)),
        Section("Herkunft der Modellurteile",
                "Modell-IDs, Abrufdaten, Aufrufbudget und nicht klassifizierte Tasten.",
                text=provenance_text(report)),
        Section("Anhang · Parameter",
                "Klasse und Behandlung jedes Parameters ohne Dev-Session.",
                text=appendix_text(report)),
    ]
