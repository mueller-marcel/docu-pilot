"""
The parameters calibrated inside each outer fold (C8), from that fold's
TRAINING sessions only.

There is no development session outside the corpus, so no parameter may be
fitted to labels of a session that is later scored. What is cheap — because it
only moves how a cached raw result is drawn — is calibrated here per fold;
what would cost model calls is frozen (see config.PARAMETERS).

    AUDIO_ANCHOR, PEAK_OFFSET_S   where an announcement's boundary lies
    REST_FULL_S                   how long a rest must be to count fully
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from docupilot.config import (
    AUDIO_MIN_PAIRS,
    PRE_S,
    REST_FULL_CANDIDATES_S,
    REST_FULL_DEFAULT_S,
    FoldParams,
    Settings,
)
from docupilot.evaluation import metrics
from docupilot.evaluation.fusion import EmptyBaseline, RuleFuser
from docupilot.segmentation import audio, events


@dataclass(frozen=True)
class AudioCalibration:
    """What the audio calibration found, including why it may have fallen back."""

    anchor: str
    offset_s: float
    n_pairs: int
    match_rate: float
    """Share of OPERATION sentences with a reference boundary in their search
    range. Low means many announcements have no action that can be placed in
    time — a low audio contribution is then explained, not an extractor bug."""
    fallback: bool
    offsets_s: dict[str, list[float]] = field(default_factory=dict)
    """Signed distances g − a_i per anchor, for the histogram in the report."""


@dataclass(frozen=True)
class Calibration:
    """Everything one fold calibrated."""

    params: FoldParams
    audio: AudioCalibration
    rest_full_scores: dict[float, float] = field(default_factory=dict)


def _nearest_in(gt: np.ndarray, lo: float, hi: float, target: float) -> float | None:
    inside = gt[(gt >= lo) & (gt <= hi)]
    if len(inside) == 0:
        return None
    return float(inside[np.argmin(np.abs(inside - target))])


def calibrate_audio(
    raws: Sequence[audio.AudioRaw], gts: Sequence[Sequence[float]]
) -> AudioCalibration:
    """
    AUDIO_ANCHOR and PEAK_OFFSET_S from the training sessions.

    For every OPERATION sentence the nearest reference boundary g in its search
    range [t_i − PRE_S, t_next] is looked up, and for both anchors (sentence
    start, sentence end) the signed distance g − a_i is collected. The anchor
    whose distances scatter less — smaller mean absolute deviation from their
    median — wins, and d is that median, sign included. Fewer than
    AUDIO_MIN_PAIRS pairs: sentence start, d = 0, and the fallback is reported.

    :param raws: the training sessions' audio raw results.
    :param gts: their reference boundaries, in the same order.
    """
    offsets: dict[str, list[float]] = {"begin": [], "end": []}
    n_operations = 0
    for raw, gt in zip(raws, gts):
        reference = np.sort(np.asarray(gt, dtype=np.float64))
        for sentence, (t_i, t_next) in zip(raw.sentences, raw.windows()):
            if sentence.category != audio.OPERATION:
                continue
            n_operations += 1
            g = _nearest_in(reference, t_i - PRE_S, max(t_next, t_i), t_i)
            if g is None:
                continue
            for anchor in offsets:
                offsets[anchor].append(g - audio.anchor_time(sentence, anchor))

    n_pairs = len(offsets["begin"])
    match_rate = n_pairs / n_operations if n_operations else 0.0
    if n_pairs < AUDIO_MIN_PAIRS:
        return AudioCalibration("begin", 0.0, n_pairs, match_rate, True, offsets)

    def spread(values: list[float]) -> float:
        median = float(np.median(values))
        return float(np.mean(np.abs(np.asarray(values) - median)))

    anchor = min(("begin", "end"), key=lambda a: (spread(offsets[a]), a != "begin"))
    return AudioCalibration(
        anchor, float(np.median(offsets[anchor])), n_pairs, match_rate, False, offsets
    )


def calibrate_rest_full(
    raws: Sequence[events.EventsRaw],
    gts: Sequence[Sequence[float]],
    durations_s: Sequence[float],
    settings: Settings,
) -> tuple[float, dict[float, float]]:
    """
    REST_FULL_S from the training sessions: for each candidate value the event
    curve is drawn and decided by A0 on the coalition {events}; the value with
    the highest macro F1 wins. A tie goes to the default (8 s), or to the
    value nearest it.

    :return: the chosen value and the macro F1 of every candidate.
    """
    rule, baseline = RuleFuser(), EmptyBaseline(float("inf"))
    scores: dict[float, float] = {}
    for value in REST_FULL_CANDIDATES_S:
        params = FoldParams(rest_full_s=value)
        f1s = [
            metrics.match(
                gt,
                rule.predict({"events": events.render(raw, settings, params)}, ["events"],
                             duration, baseline),
                settings.tau_primary_s,
            ).f1
            for raw, gt, duration in zip(raws, gts, durations_s)
        ]
        scores[value] = float(np.mean(f1s)) if f1s else 0.0
    best = max(scores.values())
    chosen = min((v for v, s in scores.items() if abs(s - best) <= 1e-12),
                 key=lambda v: abs(v - REST_FULL_DEFAULT_S))
    return chosen, scores


def calibrate(
    audio_raws: Sequence[audio.AudioRaw],
    event_raws: Sequence[events.EventsRaw],
    gts: Sequence[Sequence[float]],
    durations_s: Sequence[float],
    settings: Settings,
) -> Calibration:
    """All in-fold parameters of one fold, from its training sessions."""
    audio_cal = calibrate_audio(audio_raws, gts)
    rest_full, scores = calibrate_rest_full(event_raws, gts, durations_s, settings)
    return Calibration(
        params=FoldParams(rest_full_s=rest_full, audio_anchor=audio_cal.anchor,
                          peak_offset_s=audio_cal.offset_s),
        audio=audio_cal,
        rest_full_scores=scores,
    )
