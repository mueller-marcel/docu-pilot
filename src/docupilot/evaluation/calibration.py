"""
The parameters calibrated inside each outer fold (C8), from that fold's
TRAINING sessions only.

No parameter may be fitted to labels of a session that is later scored. The
audio parameters only move how a cached raw result is drawn. The video pair
(θ, d) is chosen without a model call too, from the SSIM scan alone; only the
changes it produces are judged afterwards. The event curves have no in-fold
parameter.

    AUDIO_PEAK_OFFSET  how long after an announcement starts its boundary lies
    VIDEO θ, d         which screen changes are candidates at all
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from docupilot.config import (
    AUDIO_MIN_PAIRS,
    VIDEO_MIN_REST_CANDIDATES_S,
    VIDEO_THETA_CANDIDATES,
    FoldParams,
    Settings,
)
from docupilot.evaluation import metrics
from docupilot.segmentation import audio, video


@dataclass(frozen=True)
class AudioCalibration:
    """What the audio calibration found, including why it may have fallen back."""

    peak_offset_s: float
    n_pairs: int
    match_rate: float
    """Share of sentences with a reference boundary within τ of their interval.
    Low means many sentences announce no action that can be placed in time —
    a low audio contribution is then explained, not an extractor bug."""
    fallback: bool
    offsets_s: list[float] = field(default_factory=list)
    """Distances boundary − sentence start, for the histogram in the report."""


@dataclass(frozen=True)
class VideoCalibration:
    """The noise threshold θ and minimum rest d one fold chose, and why."""

    theta: float
    min_rest_s: float
    f1: float
    """Pooled candidate F1 of the chosen pair on the training sessions."""
    scores: dict[tuple[float, float], float] = field(default_factory=dict)
    """Candidate F1 of every (θ, d) on the grid."""


@dataclass(frozen=True)
class Calibration:
    """Everything one fold calibrated."""

    params: FoldParams
    audio: AudioCalibration
    video: VideoCalibration


def _first_in(gt: np.ndarray, lo: float, hi: float) -> float | None:
    inside = gt[(gt >= lo) & (gt <= hi)]
    return float(inside[0]) if len(inside) else None


def calibrate_audio(
    raws: Sequence[audio.AudioRaw],
    gts: Sequence[Sequence[float]],
    settings: Settings,
) -> AudioCalibration:
    """
    AUDIO_PEAK_OFFSET from the training sessions (MA 3.2.2): how long after its
    start a sentence's bell peaks.

    The narration announces each step while or just before it is carried out,
    so a sentence leads its boundary by a fairly constant time, not by a
    constant share of its length. For every sentence [s_j, e_j], g_j is the
    first reference boundary within [s_j − τ, e_j + τ]; the offset is the
    median of max(g_j − s_j, 0). Every sentence counts, whatever the LLM said —
    the offset describes when people speak, the verdict whether a step begins.
    Fewer than AUDIO_MIN_PAIRS pairs: the sentence start (offset 0), and the
    fallback is reported.

    :param raws: the training sessions' audio raw results.
    :param gts: their reference boundaries, in the same order.
    :param settings: supplies τ.
    """
    tau = settings.tau_s
    offsets: list[float] = []
    n_sentences = 0
    for raw, gt in zip(raws, gts):
        reference = np.sort(np.asarray(gt, dtype=np.float64))
        for sentence in raw.sentences:
            n_sentences += 1
            s_j, e_j = sentence.start_s, max(sentence.end_s, sentence.start_s)
            g = _first_in(reference, s_j - tau, e_j + tau)
            if g is not None:
                offsets.append(max(g - s_j, 0.0))

    n_pairs = len(offsets)
    match_rate = n_pairs / n_sentences if n_sentences else 0.0
    if n_pairs < AUDIO_MIN_PAIRS:
        return AudioCalibration(0.0, n_pairs, match_rate, True, offsets)
    return AudioCalibration(float(np.median(offsets)), n_pairs, match_rate, False, offsets)


def calibrate_video(
    raws: Sequence[video.VideoRaw],
    gts: Sequence[Sequence[float]],
    settings: Settings,
) -> VideoCalibration:
    """
    θ and d from the training sessions, without a model call: for every pair on
    the grid the change onsets are matched 1:1 to the reference boundaries
    within τ, as in the evaluation; TP, FP and FN are pooled over the sessions
    and the pair with the highest candidate F1 wins. A tie goes to fewer
    changes, then to the longer d.

    Too coarse a pair (low θ, long d) misses boundaries; too fine a pair finds
    many changes, and every one costs a VLM call.
    """
    scores: dict[tuple[float, float], float] = {}
    best: tuple | None = None
    for theta in VIDEO_THETA_CANDIDATES:
        for d in VIDEO_MIN_REST_CANDIDATES_S:
            tp = fp = fn = 0
            for raw, gt in zip(raws, gts):
                onsets = [float(raw.times_s[c.onset])
                          for c in video.changes(raw.times_s, raw.sim, theta, d)]
                m = metrics.match(gt, onsets, settings.tau_s)
                tp, fp, fn = tp + m.tp, fp + m.fp, fn + m.fn
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            scores[(theta, d)] = f1
            key = (f1, -(tp + fp), d)
            if best is None or key > best[0]:
                best = (key, theta, d)
    assert best is not None
    return VideoCalibration(best[1], best[2], best[0][0], scores)


def calibrate(
    audio_raws: Sequence[audio.AudioRaw],
    video_raws: Sequence[video.VideoRaw],
    gts: Sequence[Sequence[float]],
    settings: Settings,
) -> Calibration:
    """All in-fold parameters of one fold, from its training sessions."""
    audio_cal = calibrate_audio(audio_raws, gts, settings)
    video_cal = calibrate_video(video_raws, gts, settings)
    return Calibration(
        params=FoldParams(audio_peak_offset_s=audio_cal.peak_offset_s,
                          video_theta=video_cal.theta, video_min_rest_s=video_cal.min_rest_s),
        audio=audio_cal,
        video=video_cal,
    )
