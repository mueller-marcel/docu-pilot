"""
The experiment itself: every modality coalition scored on every session, with
nothing that predicts a session — model, threshold, calibrated parameter —
ever having seen that session's labels.

One outer fold per session (leave-one-session-out):

    1. calibrate AUDIO_PEAK_SHARE and the video's noise threshold θ and
       minimum rest d on the training sessions (C8)
    2. draw every session's curves with those parameters — for the video this
       is where its changes are judged, once per session and (θ, d)
    3. the forest's size from the error on held-out training sessions, on the
       full coalition (MA 3.3.1)
    4. per start value and coalition: choose the threshold by an inner
       GroupKFold, fit the decider on all training sessions, predict the
       held-out one (C4)

Steps 1–3 are deterministic and run once per fold; only step 4 is repeated
per start value. A session's F1 is averaged over the start values before any
statistic, so the forest's own randomness does not widen the intervals.

`run_fold` is the whole fold and never reads the held-out session's labels;
scoring happens afterwards in `run`. Every later analysis — Shapley,
saturation, statistics — is derived from the tidy table `run` produces.

τ is set a priori (config.TOLERANCE_S); the SESOI is measured from the
reference annotation before the first run, never set (`measure_sesoi`).
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from docupilot.config import BOUNDARY_DEFINITIONS, FoldParams, Settings
from docupilot.evaluation import calibration, dataset, fusion, metrics, statistics
from docupilot.evaluation.analysis import subsets
from docupilot.evaluation.calibration import Calibration
from docupilot.recording.session import RecordingSession
from docupilot.segmentation import MODALITIES, BoundaryEvidence, extract_raws, render

# Rows are keyed by this label; the empty coalition has no modality to name.
EMPTY_SUBSET_LABEL = "{}"



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
    """One outer fold under one start value: what was calibrated, and what
    every coalition predicted."""

    held_out: str
    calibration: Calibration
    predictions: dict[str, list[float]]
    thresholds: dict[str, float]
    seed: int = 0
    n_trees: int | None = None
    """Forest size found from the held-out error (None for other deciders)."""


@dataclass
class ExperimentResult:
    """The tidy table plus what the report needs beyond it."""

    settings: Settings
    rows: list[dict] = field(default_factory=list)
    folds: list[FoldResult] = field(default_factory=list)
    """Every (held-out session, start value)."""

    @property
    def first_folds(self) -> list[FoldResult]:
        """One fold per held-out session (the first start value) — calibration
        and curves are the same for every start value."""
        return [f for f in self.folds if f.seed == self.settings.seed]


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


# ── Measured settings ─────────────────────────────────────────────────────────

def _annotation_runs(
    data: Sequence[SessionData], definition: str
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Run 1 and its blind repetition (run 2) per session, under one definition."""
    return (
        {d.name: d.gt_s(definition) for d in data if d.has(definition)},
        {d.name: d.gt_s(definition, 2) for d in data if d.has(definition, 2)},
    )


def measure_sesoi(data: Sequence[SessionData], settings: Settings) -> Settings:
    """
    The settings with the SESOI fixed from the reference annotation (MA
    3.3.3), before any run: one more boundary found, the mean of 1/n_s over
    the sessions.

    :raises ValueError: when no session has a boundary.
    """
    reference, _ = _annotation_runs(data, settings.boundary_definition)
    sesoi, source = statistics.sesoi_from_reference(reference)
    return settings.replace(sesoi=sesoi, sesoi_source=source)


# ── One fold ──────────────────────────────────────────────────────────────────

def _param_key(modality: str, params: FoldParams) -> tuple:
    """The fold parameters a modality's curve actually depends on."""
    if modality == "events":
        return ()
    if modality == "audio":
        return (params.audio_peak_offset_s,)
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
) -> list[FoldResult]:
    """
    One outer fold under every start value of `settings`. Reads the labels of
    the training sessions only — the held-out session contributes its
    recording, never its annotation.

    Calibration, curves and the forest's size are deterministic and found
    once; every start value refits the thresholds and the forests.

    :return: one result per start value, in ascending order.
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

    def prepare(members: tuple[str, ...]) -> tuple[list[fusion.Prepared], fusion.FeatureTable]:
        prepared = []
        for d, gt in zip(train, gts):
            table = _candidates(d, members, players, params, cache, pool)
            prepared.append(fusion.Prepared(
                table.times, table.select(members),
                fusion.label_candidates(table.times, gt, settings.label_tol_s), gt,
            ))
        return prepared, _candidates(test, members, players, params, cache, pool)

    tables = {
        tuple(sorted(c)): prepare(tuple(sorted(c)))
        for c in subsets(players) if c and kind != "rule"
    }

    # The forest's size (MA 3.3.1): once per fold on the full coalition's
    # training data, then the same for every coalition, start value and inner fit.
    n_trees = None
    if kind == "forest":
        n_trees, _ = fusion.tree_count(tables[tuple(sorted(players))][0], settings.seed)

    results = []
    for seed in seeds_of(settings):
        predictions: dict[str, list[float]] = {}
        thresholds: dict[str, float] = {}
        for coalition in subsets(players):
            members = tuple(sorted(coalition))
            label = subset_label(members)
            threshold = math.nan
            if not members:
                predicted = baseline.predict(test.duration_s)
            elif kind == "rule":
                predicted = fusion.RuleFuser(settings.nms_s).predict(
                    test_evidence, members, test.duration_s, baseline)
            else:
                prepared, test_table = tables[members]
                threshold = fusion.choose_threshold(
                    prepared,
                    lambda: fusion.Fuser(kind, n_estimators=n_trees, seed=seed),
                    post, tau,
                )
                final = fusion.Fuser(kind, n_estimators=n_trees, seed=seed).fit(
                    np.vstack([p.features for p in prepared]),
                    np.concatenate([p.labels for p in prepared]),
                )
                proba = final.predict_proba(test_table.select(members))
                predicted = post(test_table.times, proba, threshold)
            predictions[label] = predicted
            thresholds[label] = threshold
        results.append(FoldResult(test.name, cal, predictions, thresholds, seed, n_trees))
    return results


def seeds_of(settings: Settings) -> list[int]:
    """The start values of a run: settings.seed, seed + 1, …"""
    return [settings.seed + r for r in range(settings.n_seeds)]


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
    :param settings: the run's settings.
    :param on_progress: called as (held-out session, done, total).
    :return: one row per (session, start value, coalition), scored at τ, plus
        the folds.
    :raises ValueError: when fewer than two sessions carry the definition.
    """
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
        for fold in run_fold(data, index, settings, players, cache):
            result.folds.append(fold)
            _score_fold(result, test, fold, settings)
        if on_progress is not None:
            on_progress(test.name, index + 1, len(data))
    return result


def _score_fold(
    result: ExperimentResult, test: SessionData, fold: FoldResult, settings: Settings,
) -> None:
    """Score one (fold, start value) at τ and append its rows."""
    definition = settings.boundary_definition
    params = fold.calibration.params
    for label, predicted in fold.predictions.items():
        m = metrics.match(test.gt_s(definition), predicted, settings.tau_s)
        result.rows.append({
            "session": test.name,
            "seed": fold.seed,
            "subset": label,
            "n_modalities": len(subset_from_label(label)),
            "tau_s": settings.tau_s,
            "threshold": fold.thresholds[label],
            "audio_peak_offset_s": params.audio_peak_offset_s,
            "video_theta": params.video_theta,
            "video_min_rest_s": params.video_min_rest_s,
            "n_predicted": len(predicted),
            "tp": m.tp, "fp": m.fp, "fn": m.fn,
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
        })


# ── Reading the table ─────────────────────────────────────────────────────────

def subset_values(
    rows: Sequence[dict], tau_s: float, metric: str = "f1"
) -> dict[frozenset[str], float]:
    """
    Macro-averaged metric per coalition — the characteristic function v(S).
    Macro because sessions are the unit of the experiment; each session's
    value is first averaged over the start values.
    """
    return {k: float(np.mean(list(v.values())))
            for k, v in paired_metric(rows, tau_s, metric).items()}


def paired_metric(
    rows: Sequence[dict], tau_s: float, metric: str = "f1"
) -> dict[frozenset[str], dict[str, float]]:
    """
    Per-session value of one metric per coalition — what the statistics need —
    averaged over the start values, so the forest's randomness does not enter
    the session-to-session spread.
    """
    acc: dict[frozenset[str], dict[str, list[float]]] = {}
    for row in rows:
        if row["tau_s"] != tau_s:
            continue
        acc.setdefault(subset_from_label(row["subset"]), {}).setdefault(
            row["session"], []).append(row[metric])
    return {k: {s: float(np.mean(v)) for s, v in per.items()} for k, per in acc.items()}


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
