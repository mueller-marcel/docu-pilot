"""
Everything the evaluation reads off a recording, in the units it computes in.

The session stores boundaries in milliseconds while the modalities report
seconds — the conversion happens here and nowhere else, so a factor of 1000
cannot leak into a result. The annotation guideline's edge rules are applied
here too, once, instead of in every metric.
"""

from __future__ import annotations

from collections.abc import Sequence

from docupilot.config import GT_EDGE_S, GT_MERGE_S, PRIMARY_RUN
from docupilot.evaluation import media
from docupilot.recording.session import DEFAULT_DEFINITION, RecordingSession


def apply_edge_rules(times_s: Sequence[float], duration_s: float | None) -> list[float]:
    """
    The guideline's edge rules (N3): boundaries in the first or last GT_EDGE_S
    seconds are dropped, and boundaries closer than GT_MERGE_S are merged into
    their mean — a double click on the annotation button is one boundary.

    :param duration_s: length of the recording; None skips the end rule.
    """
    kept = sorted(
        t for t in times_s
        if t >= GT_EDGE_S and (duration_s is None or t <= duration_s - GT_EDGE_S)
    )
    merged: list[list[float]] = []
    for t in kept:
        if merged and t - merged[-1][-1] < GT_MERGE_S:
            merged[-1].append(t)
        else:
            merged.append([t])
    return [sum(group) / len(group) for group in merged]


def ground_truth_s(
    session: RecordingSession,
    definition: str = DEFAULT_DEFINITION,
    run: int = PRIMARY_RUN,
    duration_s: float | None = None,
) -> list[float]:
    """
    The annotated boundaries of one definition and run in seconds, ascending,
    after the edge rules.

    :param session: the recording; its ground_truth.json holds milliseconds.
    :param definition: "begin" (primary) or "end" (sensitivity).
    :param run: 1 for the annotation, 2 for its blind repetition.
    :param duration_s: length of the recording, for the end rule.
    :return: boundary timestamps in seconds.
    """
    raw = [t_ms / 1000.0 for t_ms, _ in session.ground_truth_markers(definition, run)]
    return apply_edge_rules(raw, duration_s)


def duration_s(session: RecordingSession) -> float:
    """
    Length of the recording in seconds, read from the container.

    Needed for the candidate grid, the empty-coalition baseline and the chance
    level: all of them spread over the whole recording, so a wrong length would
    silently move them.

    :param session: the recording; its .recording_path points at the MP4.
    :return: duration in seconds.
    :raises RuntimeError: when the duration cannot be read.
    """
    info = media.probe(session.recording_path)
    try:
        return info.duration_s
    except RuntimeError:
        raise RuntimeError(
            f"Dauer von {session.recording_path} nicht lesbar — ffprobe lieferte "
            f"{info.duration_raw!r}."
        ) from None
