"""
The experiment itself: every modality coalition scored on every session, with
nothing that predicts a session — model, threshold, calibrated parameter —
ever having seen that session's labels.

One outer fold per session (leave-one-session-out):

    1. calibrate AUDIO_ANCHOR, PEAK_OFFSET_S and the video's
       noise threshold θ and minimum rest d on the training sessions (C8)
    2. draw every session's curves with those parameters — for the video this
       is where its changes are judged, once per session and (θ, d)
    3. per coalition: choose the threshold by an inner GroupKFold, fit the
       decider on all training sessions, predict the held-out one (C4)

`run_fold` is the whole fold and never reads the held-out session's labels;
scoring happens afterwards in `run`. Every later analysis — Shapley,
saturation, statistics — is derived from the tidy table `run` produces.

τ and the SESOI are measured, never set: `measure_tau` before the first run,
`measure_sesoi` from the primary run's predictions before any statistics.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from docupilot.config import BOUNDARY_DEFINITIONS, TAU_SWEEP_S, FoldParams, Settings
from docupilot.evaluation import calibration, dataset, fusion, metrics, reliability
from docupilot.evaluation.analysis import subsets
from docupilot.evaluation.calibration import Calibration
from docupilot.recording.session import RecordingSession
from docupilot.segmentation import MODALITIES, BoundaryEvidence, extract_raws, render

# Rows are keyed by this label; the empty coalition has no modality to name.
EMPTY_SUBSET_LABEL = "{}"

# Settings that change a modality's RAW result; a variant differing in one of
# them needs its own extraction (burst pause: free; ASR alignment: model calls).
RAW_SETTINGS: tuple[str, ...] = ("burst_pause_s", "asr_alignment")


@dataclass(frozen=True)
class SessionData:
    """One recording with everything the experiment needs, extracted once."""

    name: str
    duration_s: float
    raws: Mapping[str, Any]
    annotations: Mapping[tuple[str, int], list[float]]
    """(definition, run) -> boundaries in seconds, edge rules applied."""

    def gt_s(self, definition: str = "begin", run: int = 1) -> list[float]:
        return list(self.annotations.get((definition, run), []))

    def has(self, definition: str, run: int = 1) -> bool:
        return (definition, run) in self.annotations


@dataclass(frozen=True)
class FoldResult:
    """One outer fold: what was calibrated, and what every coalition predicted."""

    held_out: str
    calibration: Calibration
    predictions: dict[str, list[float]]
    thresholds: dict[str, float]
    thresholds_oob: dict[str, float]
    evidence: dict[str, BoundaryEvidence]
    """The held-out session's curves as this fold drew them."""


@dataclass
class ExperimentResult:
    """The tidy table plus what the report needs beyond it."""

    settings: Settings
    rows: list[dict] = field(default_factory=list)
    folds: list[FoldResult] = field(default_factory=list)

    @property
    def label(self) -> str:
        return variant_label(self.settings)

    def predictions(self, subset: str) -> dict[str, list[float]]:
        """Per session, what one coalition predicted (label from subset_label)."""
        return {f.held_out: f.predictions[subset] for f in self.folds}


def variant_label(settings: Settings) -> str:
    """A short name of a run's switches, for tables and CSV rows."""
    return (f"{settings.decider}·{settings.candidate_pool}·{settings.boundary_definition}"
            f"·ev={settings.event_channels}·pause={settings.burst_pause_s:g}")


def subset_label(subset: Sequence[str] | frozenset[str]) -> str:
    """The row key for a coalition: its modalities sorted and joined."""
    return "+".join(sorted(subset)) or EMPTY_SUBSET_LABEL


def subset_from_label(label: str) -> frozenset[str]:
    """Inverse of `subset_label`."""
    return frozenset() if label == EMPTY_SUBSET_LABEL else frozenset(label.split("+"))


# ── Loading ───────────────────────────────────────────────────────────────────

def load(
    directories: Sequence[Path],
    settings: Settings | None = None,
    use_cache: bool = True,
    on_progress: Callable[[str, int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> list[SessionData]:
    """
    Extract every modality's raw result once per session and keep it in memory.

    The audio model calls happen here — once per session, not per coalition,
    fold or calibrated parameter. The video's VLM calls follow in the folds,
    once per session and chosen (θ, d); a pair chosen again is cached.

    :param on_progress: called as (session name, done, total).
    :param is_cancelled: polled between and inside sessions; True stops early
        and returns what was extracted so far.
    :raises RuntimeError: when a modality fails — an empty lane would silently
        look like a modality that found nothing.
    """
    settings = settings or Settings()
    out: list[SessionData] = []
    for i, directory in enumerate(directories, start=1):
        if is_cancelled is not None and is_cancelled():
            return out
        session = RecordingSession.from_directory(directory)
        raws: dict[str, Any] = {}
        errors: dict[str, str] = {}
        extract_raws(session, raws.__setitem__, errors.__setitem__, settings,
                     is_cancelled=is_cancelled, use_cache=use_cache)
        if is_cancelled is not None and is_cancelled():
            return out
        if errors:
            raise RuntimeError(
                f"{directory.name}: Modalität(en) fehlgeschlagen — "
                + "; ".join(f"{m}: {msg.splitlines()[0]}" for m, msg in errors.items())
            )
        duration = dataset.duration_s(session)
        annotations = {
            (definition, run): dataset.ground_truth_s(session, definition, run, duration)
            for definition in BOUNDARY_DEFINITIONS
            for run in session.annotation_runs(definition)
        }
        out.append(SessionData(directory.name, duration, raws, annotations))
        if on_progress is not None:
            on_progress(directory.name, i, len(directories))
    return out


def needs_extraction(primary: Settings, variant: Settings) -> bool:
    """True when a variant's raw results differ from the primary run's."""
    return any(getattr(primary, n) != getattr(variant, n) for n in RAW_SETTINGS)


# ── Measured settings ─────────────────────────────────────────────────────────

def _annotation_runs(
    data: Sequence[SessionData], definition: str
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Run 1 and its blind repetition (run 2) per session, under one definition."""
    return (
        {d.name: d.gt_s(definition) for d in data if d.has(definition)},
        {d.name: d.gt_s(definition, 2) for d in data if d.has(definition, 2)},
    )


def measure_tau(data: Sequence[SessionData], settings: Settings) -> Settings:
    """
    The settings with τ measured from the corpus' repeated annotation (formula
    3.1), before any run.

    :raises ValueError: when the corpus has no usable repetition.
    """
    tau, source = reliability.measure_tau(*_annotation_runs(data, settings.boundary_definition))
    return settings.replace(tau_primary_s=tau, tau_source=source)


def measure_sesoi(data: Sequence[SessionData], primary: ExperimentResult) -> Settings:
    """
    The primary run's settings with the SESOI measured from its full model's
    predictions on the retest sessions. Stored on `primary` as well, so the
    statistics of the primary run use it; variants are made from the result.

    :raises ValueError: when the retest gives no positive SESOI.
    """
    s = primary.settings
    sesoi, source = reliability.measure_sesoi(
        *_annotation_runs(data, s.boundary_definition),
        primary.predictions(subset_label(MODALITIES)), s.tau_s, s.sesoi_rule,
    )
    primary.settings = s.replace(sesoi=sesoi, sesoi_source=source)
    return primary.settings


# ── One fold ──────────────────────────────────────────────────────────────────

def _param_key(modality: str, params: FoldParams) -> tuple:
    """The fold parameters a modality's curve actually depends on."""
    if modality == "events":
        return ()
    if modality == "audio":
        return (params.audio_anchor, params.peak_offset_s)
    return (params.video_theta, params.video_min_rest_s)


class FeatureCache:
    """
    Curves and feature blocks shared across folds. Most folds calibrate the
    same parameters, so a block is computed once per (session, modality,
    parameters, candidates).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._evidence: dict[tuple, BoundaryEvidence] = {}
        self._blocks: dict[tuple, tuple[np.ndarray, list[str]]] = {}
        self._grids: dict[str, np.ndarray] = {}

    def evidence(self, session: SessionData, modality: str, params: FoldParams) -> BoundaryEvidence:
        key = (session.name, modality, _param_key(modality, params))
        if key not in self._evidence:
            self._evidence[key] = render(modality, session.raws[modality], self._settings, params)
        return self._evidence[key]

    def grid(self, session: SessionData) -> np.ndarray:
        if session.name not in self._grids:
            self._grids[session.name] = fusion.grid_candidates(
                session.duration_s, self._settings.cand_spacing_s)
        return self._grids[session.name]

    def table(
        self, session: SessionData, cand: np.ndarray, cand_key: tuple,
        modalities: Sequence[str], params: FoldParams,
    ) -> fusion.FeatureTable:
        blocks = []
        for m in modalities:
            key = (session.name, m, _param_key(m, params), cand_key)
            if key not in self._blocks:
                self._blocks[key] = fusion.modality_features(
                    self.evidence(session, m, params), cand, self._settings.tau_s)
            blocks.append((m, *self._blocks[key]))
        return fusion.FeatureTable.from_blocks(cand, blocks)


def _candidates(
    session: SessionData, coalition: Sequence[str], players: Sequence[str],
    params: FoldParams, cache: FeatureCache, pool: str,
) -> fusion.FeatureTable:
    """The candidates of one session under a pool, described by the coalition only."""
    if pool == "grid":
        return cache.table(session, cache.grid(session), ("grid",), coalition, params)
    source = players if pool == "union" else coalition
    evidence = {m: cache.evidence(session, m, params) for m in source}
    cand = fusion.candidate_times(evidence, source)
    key = (pool, tuple(sorted(source)), tuple(_param_key(m, params) for m in sorted(source)))
    return cache.table(session, cand, key, coalition, params)


def run_fold(
    data: Sequence[SessionData],
    held_out: int,
    settings: Settings,
    players: Sequence[str] = MODALITIES,
    cache: FeatureCache | None = None,
) -> FoldResult:
    """
    One outer fold. Reads the labels of the training sessions only — the
    held-out session contributes its recording, never its annotation.
    """
    cache = cache or FeatureCache(settings)
    definition = settings.boundary_definition
    tau = settings.tau_s
    train = [d for i, d in enumerate(data) if i != held_out]
    test = data[held_out]
    gts = [d.gt_s(definition) for d in train]

    cal = calibration.calibrate(
        [d.raws["audio"] for d in train], [d.raws["video"] for d in train], gts, settings,
    )
    params = cal.params
    baseline = fusion.EmptyBaseline.fit([d.duration_s for d in train], [len(g) for g in gts])
    test_evidence = {m: cache.evidence(test, m, params) for m in players}

    pool = settings.candidate_pool
    kind = settings.decider

    def post(times: np.ndarray, proba: np.ndarray, threshold: float) -> list[float]:
        return fusion.decide(times, proba, threshold, settings.nms_s, peaks=pool == "grid")

    predictions: dict[str, list[float]] = {}
    thresholds: dict[str, float] = {}
    thresholds_oob: dict[str, float] = {}
    for coalition in subsets(players):
        members = tuple(sorted(coalition))
        label = subset_label(members)
        threshold = threshold_oob = math.nan
        if not members:
            predicted = baseline.predict(test.duration_s)
        elif kind == "rule":
            predicted = fusion.RuleFuser(settings.nms_s).predict(
                test_evidence, members, test.duration_s, baseline)
        else:
            prepared = []
            for d, gt in zip(train, gts):
                table = _candidates(d, members, players, params, cache, pool)
                prepared.append(fusion.Prepared(
                    table.times, table.select(members),
                    fusion.label_candidates(table.times, gt, settings.label_tol_s), gt,
                ))
            test_table = _candidates(test, members, players, params, cache, pool)

            threshold = fusion.choose_threshold(
                prepared,
                lambda: fusion.Fuser(kind, n_estimators=fusion.trees_inner(kind),
                                     seed=settings.seed),
                post, tau,
            )
            final = fusion.Fuser(kind, seed=settings.seed, oob=kind == "forest").fit(
                np.vstack([p.features for p in prepared]),
                np.concatenate([p.labels for p in prepared]),
            )
            if final.oob_proba is not None:
                threshold_oob = fusion.oob_threshold(prepared, final.oob_proba, post, tau)
            proba = final.predict_proba(test_table.select(members))
            predicted = post(test_table.times, proba, threshold)
        predictions[label] = predicted
        thresholds[label] = threshold
        thresholds_oob[label] = threshold_oob

    return FoldResult(test.name, cal, predictions, thresholds, thresholds_oob, test_evidence)


# ── The whole experiment ──────────────────────────────────────────────────────

def usable(data: Sequence[SessionData], definition: str) -> list[SessionData]:
    """The sessions annotated under a definition (run 1)."""
    return [d for d in data if d.has(definition)]


def run(
    data: Sequence[SessionData],
    settings: Settings,
    players: Sequence[str] = MODALITIES,
    on_progress: Callable[[str, int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> ExperimentResult:
    """
    Score every coalition on every session, leaving that session out.

    :param data: the corpus; at least two sessions annotated under the
        settings' boundary definition.
    :param settings: with τ measured (measure_tau).
    :param on_progress: called as (held-out session, done, total).
    :return: one row per (session, coalition, tolerance), plus the folds; the
        tolerances are the measured τ and the robustness sweep TAU_SWEEP_S.
    :raises ValueError: when fewer than two sessions carry the definition, or
        τ was not measured.
    """
    taus_s = sorted({settings.tau_s, *TAU_SWEEP_S})
    definition = settings.boundary_definition
    data = usable(data, definition)
    if len(data) < 2:
        raise ValueError(
            "Leave-one-session-out braucht mindestens zwei Sessions mit Grenzen der "
            f"Definition {definition!r} — mit einer gäbe es keine Trainingsdaten, die "
            "die bewertete Session nicht enthalten."
        )

    result = ExperimentResult(settings)
    cache = FeatureCache(settings)
    for index, test in enumerate(data):
        if is_cancelled is not None and is_cancelled():
            break
        fold = run_fold(data, index, settings, players, cache)
        result.folds.append(fold)
        params = fold.calibration.params
        video = test.raws["video"].judged.get((params.video_theta, params.video_min_rest_s))
        audio_cal = fold.calibration.audio
        for label, predicted in fold.predictions.items():
            for tau in taus_s:
                m = metrics.match(test.gt_s(definition), predicted, tau)
                result.rows.append({
                    "session": test.name,
                    "variant": result.label,
                    "decider": settings.decider,
                    "pool": settings.candidate_pool,
                    "definition": definition,
                    "event_channels": settings.event_channels,
                    "subset": label,
                    "n_modalities": len(subset_from_label(label)),
                    "tau_s": tau,
                    "threshold": fold.thresholds[label],
                    "threshold_oob": fold.thresholds_oob[label],
                    "audio_anchor": fold.calibration.params.audio_anchor,
                    "peak_offset_s": fold.calibration.params.peak_offset_s,
                    "audio_pairs": audio_cal.n_pairs,
                    "audio_match_rate": audio_cal.match_rate,
                    "audio_fallback": audio_cal.fallback,
                    "video_theta": params.video_theta,
                    "video_min_rest_s": params.video_min_rest_s,
                    "video_candidate_f1": fold.calibration.video.f1,
                    "video_changes": getattr(video, "n_changes", 0),
                    "video_calls": getattr(video, "calls", 0),
                    "video_limit_hit": getattr(video, "limit_hit", False),
                    "n_predicted": len(predicted),
                    "tp": m.tp, "fp": m.fp, "fn": m.fn,
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1": m.f1,
                })
        if on_progress is not None:
            on_progress(test.name, index + 1, len(data))
    return result


# ── Reading the table ─────────────────────────────────────────────────────────

def subset_values(
    rows: Sequence[dict], tau_s: float, metric: str = "f1"
) -> dict[frozenset[str], float]:
    """
    Macro-averaged metric per coalition — the characteristic function v(S).
    Macro because sessions are the unit of the experiment.
    """
    per_subset: dict[frozenset[str], list[float]] = {}
    for row in rows:
        if row["tau_s"] != tau_s:
            continue
        per_subset.setdefault(subset_from_label(row["subset"]), []).append(row[metric])
    return {k: float(np.mean(v)) for k, v in per_subset.items()}


def paired_metric(
    rows: Sequence[dict], tau_s: float, metric: str = "f1"
) -> dict[frozenset[str], dict[str, float]]:
    """Per-session value of one metric per coalition — what the statistics need."""
    out: dict[frozenset[str], dict[str, float]] = {}
    for row in rows:
        if row["tau_s"] != tau_s:
            continue
        out.setdefault(subset_from_label(row["subset"]), {})[row["session"]] = row[metric]
    return out


def taus_in(rows: Sequence[dict]) -> list[float]:
    """The tolerances a result table was scored at, ascending."""
    return sorted({row["tau_s"] for row in rows})


def write_csv(rows: Sequence[dict], path: Path) -> None:
    """Write the tidy table; figures for the thesis are generated from this file."""
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
