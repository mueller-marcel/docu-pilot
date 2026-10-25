"""
How a set of modalities becomes one prediction: fix the moments that are
decided about, describe each with what those modalities say about it, then
decide.

Everything here reads only the modalities of the coalition it was given. A
single look at an excluded modality would make the Shapley values measure the
leak — and "where to look" is information too, which is why the primary
candidate pool is a neutral grid, identical for every coalition.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import rankdata

from docupilot.config import (
    FEATURE_WINDOWS_TAU,
    FOREST_MAX_SAMPLES,
    FOREST_MIN_SAMPLES_LEAF,
    FOREST_TREES_MAX,
    FOREST_TREES_STEP,
    GRID_HZ,
    INNER_FOLDS,
    THRESHOLD_GRID,
    TREES_HOLDOUT_SHARE,
    TREES_STABLE_STEPS,
    TREES_STABLE_TOL,
)
from docupilot.evaluation import metrics
from docupilot.segmentation.evidence import BOUNDARY_THRESHOLD, BoundaryEvidence

# The same recipe for every curve: an asymmetric one would move the Shapley
# values by design effort, not by information.
FEATURE_NAMES: tuple[str, ...] = (
    "point",
    *(f"max_{k:g}tau" for k in FEATURE_WINDOWS_TAU),
    "rank",
)
FEATURES_PER_CURVE = len(FEATURE_NAMES)

Evidence = Mapping[str, BoundaryEvidence]


def grid_candidates(duration_s: float, spacing_s: float) -> np.ndarray:
    """
    Candidates every `spacing_s` (τ/5) over the recording — the "grid" pool.

    Depends on nothing but the recording's length and τ, so it is bit-identical
    for all eight coalitions of a session and carries no modality information.
    """
    return np.arange(0.0, max(duration_s, 0.0), spacing_s, dtype=np.float64)


def candidate_times(evidence: Evidence, subset: Sequence[str]) -> np.ndarray:
    """
    The moments the given modalities propose: every local maximum of their score
    curves, pooled and sorted — the "own" and "union" pools (sensitivity only).

    :param subset: which modalities may contribute — no others are read.
    :return: candidate timestamps in seconds, ascending. Empty for an empty subset.
    """
    times: list[float] = []
    for modality in subset:
        ev = evidence[modality]
        if len(ev.score) < 3:
            continue
        peaks, _ = find_peaks(ev.score)
        times.extend(float(ev.times_s[i]) for i in peaks)
    return np.sort(np.asarray(times, dtype=np.float64))


def _window_max(times_s: np.ndarray, curve: np.ndarray, cand: np.ndarray, window_s: float) -> np.ndarray:
    """Highest value within ±window_s of each candidate; 0 where the window is empty."""
    lo = np.searchsorted(times_s, cand - window_s, side="left")
    hi = np.searchsorted(times_s, cand + window_s, side="right")
    # reduceat over the interleaved bounds takes the max of curve[lo:hi]; the
    # sentinel keeps an index equal to len(curve) valid.
    padded = np.append(curve.astype(np.float64), -np.inf)
    bounds = np.empty(2 * len(cand), dtype=np.int64)
    bounds[0::2], bounds[1::2] = lo, hi
    out = np.maximum.reduceat(padded, bounds)[0::2]
    return np.where(hi > lo, out, 0.0)


def curve_features(
    times_s: np.ndarray, curve: np.ndarray, cand: np.ndarray, tau_s: float
) -> np.ndarray:
    """One curve's block: (n_candidates, FEATURES_PER_CURVE)."""
    n = len(cand)
    block = np.zeros((n, FEATURES_PER_CURVE), dtype=np.float64)
    if n == 0 or len(times_s) == 0:
        return block
    block[:, 0] = np.interp(cand, times_s, curve, left=0.0, right=0.0)
    for column, k in enumerate(FEATURE_WINDOWS_TAU, start=1):
        block[:, column] = _window_max(times_s, curve, cand, k * tau_s)
    rank_source = _window_max(times_s, curve, cand, tau_s)
    # Average ranks scaled to (0, 1]: ties (many zeros) share a rank instead of
    # being ordered by position, which would leak the candidate index.
    block[:, -1] = rankdata(rank_source, method="average") / n
    return block


def modality_features(
    ev: BoundaryEvidence, cand: np.ndarray, tau_s: float
) -> tuple[np.ndarray, list[str]]:
    """
    One modality's feature block over all its curves (one, or the event
    modality's sub-curves), with the column names.
    """
    blocks, names = [], []
    for channel, curve in ev.feature_curves().items():
        blocks.append(curve_features(ev.times_s, curve, cand, tau_s))
        names.extend(f"{channel}.{f}" for f in FEATURE_NAMES)
    return np.hstack(blocks) if blocks else np.zeros((len(cand), 0)), names


@dataclass(frozen=True)
class FeatureTable:
    """The features of one session's candidates, each column owned by one modality."""

    times: np.ndarray
    matrix: np.ndarray
    owners: tuple[str, ...]
    names: tuple[str, ...]

    @classmethod
    def from_blocks(
        cls, cand: np.ndarray, blocks: Sequence[tuple[str, np.ndarray, Sequence[str]]]
    ) -> FeatureTable:
        """Assemble (modality, block, column names) blocks, in the given order."""
        owners = tuple(m for m, block, _ in blocks for _ in range(block.shape[1]))
        names = tuple(f"{m}.{n}" for m, _, block_names in blocks for n in block_names)
        matrix = (np.hstack([block for _, block, _ in blocks]) if blocks
                  else np.zeros((len(cand), 0), dtype=np.float64))
        return cls(cand, matrix, owners, names)

    def columns(self, coalition: Sequence[str]) -> list[int]:
        """The column indices a coalition may read — only its own modalities'."""
        wanted = set(coalition)
        return [i for i, owner in enumerate(self.owners) if owner in wanted]

    def select(self, coalition: Sequence[str]) -> np.ndarray:
        columns = self.columns(coalition)
        assert all(self.owners[i] in coalition for i in columns)
        return self.matrix[:, columns]


def label_candidates(times: np.ndarray, gt_s: Sequence[float], tol_s: float) -> np.ndarray:
    """
    True for every candidate within `tol_s` of an annotated boundary.

    Deliberately not one-to-one: this is the training signal, not the score. On
    the τ/5 grid eleven candidates around each boundary are positive.
    """
    if len(times) == 0:
        return np.zeros(0, dtype=bool)
    gt = np.sort(np.asarray(gt_s, dtype=np.float64))
    if len(gt) == 0:
        return np.zeros(len(times), dtype=bool)
    index = np.searchsorted(gt, times)
    right = np.abs(gt[np.clip(index, 0, len(gt) - 1)] - times)
    left = np.abs(gt[np.clip(index - 1, 0, len(gt) - 1)] - times)
    return np.minimum(left, right) <= tol_s


def suppress(times: np.ndarray, scores: np.ndarray, radius_s: float) -> list[float]:
    """
    Non-maximum suppression: keep the strongest candidate in each neighbourhood.

    :return: the surviving timestamps, ascending.
    """
    kept: list[float] = []
    for i in np.argsort(scores, kind="stable")[::-1]:
        t = float(times[i])
        if all(abs(t - k) > radius_s for k in kept):
            kept.append(t)
    return sorted(kept)


def decide(
    times: np.ndarray, proba: np.ndarray, threshold: float, radius_s: float, *, peaks: bool
) -> list[float]:
    """
    The post-processing — identical for choosing the threshold and for the test
    prediction: p ≥ threshold, then the local maxima of p along time (on
    the grid, where neighbouring candidates are the same moment), then
    non-maximum suppression within `radius_s`.

    :param peaks: pick local maxima first; True for the grid pool, whose
        candidates form a time series. Sparse pools go straight to suppression.
    """
    if len(times) == 0:
        return []
    above = np.where(proba >= threshold, proba, 0.0)
    if peaks:
        # Zero padding lets a maximum on the first or last candidate count.
        index, _ = find_peaks(np.concatenate(([0.0], above, [0.0])))
        index = index - 1
        index = index[above[index] > 0]
    else:
        index = np.flatnonzero(proba >= threshold)
    return suppress(times[index], proba[index], radius_s)


@dataclass(frozen=True)
class EmptyBaseline:
    """
    What can be predicted with no modality at all: boundaries at the mean
    segment length of the training sessions, at fixed distances.

    Reads no modality — only the durations and boundary counts of the TRAINING
    sessions. Without it v(∅) = 0, and Shapley would distribute this
    information-free performance over the modalities, inflating the weak ones.
    """

    spacing_s: float

    @classmethod
    def fit(cls, durations_s: Sequence[float], boundary_counts: Sequence[int]) -> EmptyBaseline:
        segments = sum(n + 1 for n in boundary_counts)
        return cls(float(sum(durations_s)) / segments if segments else float("inf"))

    def predict(self, duration_s: float) -> list[float]:
        if not np.isfinite(self.spacing_s) or self.spacing_s <= 0:
            return []
        return [float(t) for t in np.arange(self.spacing_s, duration_s, self.spacing_s)]


class RuleFuser:
    """
    The untrained baseline A0: the maximum of the coalition's score curves,
    its peaks of at least 0.5, at least NMS_S apart — the NMS radius of the
    trained deciders. No training, no threshold.

    Shows whether a finding rests on the modalities or only on the learned
    combiner. The event score is already the weighted maximum of its
    sub-curves (1 : 1 : 0.5).
    """

    def __init__(self, distance_s: float) -> None:
        self.distance_s = distance_s

    def predict(
        self,
        evidence: Evidence,
        coalition: Sequence[str],
        duration_s: float,
        baseline: EmptyBaseline,
    ) -> list[float]:
        if not coalition:
            return baseline.predict(duration_s)
        t = np.arange(0.0, max(duration_s, 0.0), 1.0 / GRID_HZ)
        if len(t) == 0:
            return []
        s = np.max([
            np.interp(t, evidence[m].times_s, evidence[m].score, left=0.0, right=0.0)
            if len(evidence[m].times_s) else np.zeros(len(t))
            for m in coalition
        ], axis=0)
        index, _ = find_peaks(
            np.concatenate(([0.0], s, [0.0])),
            height=BOUNDARY_THRESHOLD,
            distance=max(1, int(self.distance_s * GRID_HZ)),
        )
        return [float(t[i - 1]) for i in index]


class _Constant:
    """A fold whose rows carry one label only: the answer is that label."""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.column_stack([1.0 - np.full(len(features), self.value),
                                np.full(len(features), self.value)])


class Fuser:
    """
    A trained decider over a coalition's features: "forest" (random forest) or
    "logreg" (standardised logistic regression), both with balanced class
    weights because the grid is ~1:15 imbalanced.
    """

    def __init__(self, kind: str, *, n_estimators: int | None = None, seed: int = 0) -> None:
        """:param n_estimators: the fold's forest size (tree_count)."""
        self.kind = kind
        self._n_estimators = n_estimators or FOREST_TREES_MAX
        self._seed = seed
        self._model = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> Fuser:
        classes = np.unique(labels)
        if len(classes) < 2:
            value = float(classes[0]) if len(classes) else 0.0
            self._model = _Constant(value)
            return self

        if self.kind == "forest":
            from sklearn.ensemble import RandomForestClassifier

            self._model = RandomForestClassifier(
                n_estimators=self._n_estimators,
                class_weight="balanced",
                min_samples_leaf=FOREST_MIN_SAMPLES_LEAF,
                max_samples=FOREST_MAX_SAMPLES,
                random_state=self._seed,
                n_jobs=-1,
            ).fit(features, labels)
        elif self.kind == "logreg":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            self._model = make_pipeline(
                StandardScaler(),
                LogisticRegression(class_weight="balanced", max_iter=2000,
                                   random_state=self._seed),
            ).fit(features, labels)
        else:
            raise ValueError(f"Unknown decider {self.kind!r}")
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("predict_proba() called before fit()")
        if len(features) == 0:
            return np.zeros(0, dtype=np.float64)
        return self._model.predict_proba(features)[:, 1]


@dataclass(frozen=True)
class Prepared:
    """One training session as the threshold search sees it."""

    times: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    gt_s: Sequence[float]


def best_threshold(
    predictions: Sequence[tuple[np.ndarray, np.ndarray, Sequence[float]]],
    post: Callable[[np.ndarray, np.ndarray, float], list[float]],
    tau_s: float,
    grid: Sequence[float] = THRESHOLD_GRID,
) -> float:
    """
    The threshold that maximises the mean per-session F1 of the post-processed
    predictions. Macro, because the attributed quantity is a macro average.
    Ties go to the value nearest 0.5, so a flat optimum is not pulled to an
    extreme by grid order.

    :param predictions: (times, probabilities, ground truth) per session.
    :param post: the post-processing, (times, proba, threshold) -> boundaries.
    """
    scores = {
        float(t): float(np.mean([
            metrics.match(gt, post(times, proba, t), tau_s).f1
            for times, proba, gt in predictions
        ])) if predictions else 0.0
        for t in grid
    }
    best = max(scores.values())
    return min((t for t, s in scores.items() if abs(s - best) <= 1e-12),
               key=lambda t: (abs(t - 0.5), t))


def choose_threshold(
    sessions: Sequence[Prepared],
    make_fuser: Callable[[], Fuser],
    post: Callable[[np.ndarray, np.ndarray, float], list[float]],
    tau_s: float,
    n_splits: int = INNER_FOLDS,
) -> float:
    """
    The decision threshold of one (outer fold, coalition), chosen by an INNER
    GroupKFold over the training sessions.

    Every probability the threshold is chosen on comes from a model that did
    not see that session. Out-of-bag votes would not do: a candidate's temporal
    neighbours from the same session sit in the same tree's training data and
    are strongly correlated, so OOB is optimistic — exactly what grouping by
    session is meant to prevent.

    :param sessions: the outer fold's training sessions only.
    :param make_fuser: a fresh, cheaper decider for each inner fit.
    :param post: the same post-processing the test prediction uses.
    """
    from sklearn.model_selection import GroupKFold

    n = len(sessions)
    if n < 2:
        return 0.5
    index = np.arange(n)
    oof: dict[int, np.ndarray] = {}
    for inner_train, inner_val in GroupKFold(n_splits=min(n_splits, n)).split(index, groups=index):
        fuser = make_fuser().fit(
            np.vstack([sessions[i].features for i in inner_train]),
            np.concatenate([sessions[i].labels for i in inner_train]),
        )
        for i in inner_val:
            oof[int(i)] = fuser.predict_proba(sessions[i].features)
    return best_threshold(
        [(sessions[i].times, oof[i], sessions[i].gt_s) for i in range(n)], post, tau_s
    )


def tree_count(sessions: Sequence[Prepared], seed: int = 0) -> tuple[int, list[float]]:
    """
    The number of trees after which the forest's error has settled
    (Hastie et al. 2009, p. 593): high with few trees, then falling; once it
    stays flat there are enough. Grown with warm start, so the whole search
    costs about one forest of the final size.

    The error is measured on held-out training SESSIONS (TREES_HOLDOUT_SHARE
    of them, drawn with `seed`) rather than on out-of-bag rows — see
    config.FOREST_TREES_STEP for why OOB is biased on the grid — as the Brier
    score, which needs no decision threshold.

    :param sessions: the outer fold's training sessions only.
    :return: (tree count, holdout error after each step).
    """
    from sklearn.ensemble import RandomForestClassifier

    order = np.random.default_rng(seed).permutation(len(sessions))
    n_val = max(1, int(round(TREES_HOLDOUT_SHARE * len(sessions))))
    val, fit = order[:n_val], order[n_val:]
    if len(fit) == 0:
        return FOREST_TREES_STEP, []
    x_fit = np.vstack([sessions[i].features for i in fit])
    y_fit = np.concatenate([sessions[i].labels for i in fit])
    x_val = np.vstack([sessions[i].features for i in val])
    y_val = np.concatenate([sessions[i].labels for i in val]).astype(np.float64)
    if len(np.unique(y_fit)) < 2 or len(y_val) == 0:
        return FOREST_TREES_STEP, []

    forest = RandomForestClassifier(
        n_estimators=FOREST_TREES_STEP, class_weight="balanced",
        min_samples_leaf=FOREST_MIN_SAMPLES_LEAF, max_samples=FOREST_MAX_SAMPLES,
        random_state=seed, warm_start=True, n_jobs=-1,
    )
    errors: list[float] = []
    flat = 0
    n = FOREST_TREES_STEP
    while True:
        forest.set_params(n_estimators=n)
        with warnings.catch_warnings():
            # warm_start keeps the class weights of the first fit — same data,
            # so the same weights; sklearn warns regardless.
            warnings.simplefilter("ignore", UserWarning)
            forest.fit(x_fit, y_fit)
        p = forest.predict_proba(x_val)[:, 1]
        errors.append(float(np.mean((p - y_val) ** 2)))
        if len(errors) >= 2 and abs(errors[-1] - errors[-2]) < TREES_STABLE_TOL:
            flat += 1
        else:
            flat = 0
        if flat >= TREES_STABLE_STEPS or n >= FOREST_TREES_MAX:
            return n, errors
        n += FOREST_TREES_STEP
