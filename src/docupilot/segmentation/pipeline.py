"""
The one entry point into segmentation: run every modality over one session.

Every modality module offers the same two stages:

    extract_raw(session, settings, *, use_cache, on_progress, is_cancelled)
        -> raw result (the expensive part: decoding, model calls)
    render(raw, settings, params) -> BoundaryEvidence (cheap; the video judges
        its changes here, once per fold parameters, because they decide them)

A fourth modality means a module with `MODALITY`, `raw_key`, `extract_raw` and
`render`, named in _EXTRACTORS. No caller changes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from docupilot.config import FoldParams, Settings
from docupilot.recording.session import RecordingSession
from docupilot.segmentation import audio, events, store, video
from docupilot.segmentation.evidence import BoundaryEvidence

# Cheapest first, slowest last, so a caller rendering results as they arrive
# fills the screen in that order.
_EXTRACTORS = (events, video, audio)

MODALITIES: tuple[str, ...] = tuple(m.MODALITY for m in _EXTRACTORS)

_BY_NAME = {m.MODALITY: m for m in _EXTRACTORS}

# Version of how a lane is drawn (evidence.py). Bump it when the curve shapes
# change, so a lane stored with the old shapes is a miss rather than served.
_CURVE_FORMAT = 2


def render(
    modality: str, raw: Any, settings: Settings, params: FoldParams | None = None
) -> BoundaryEvidence:
    """Draw one modality's raw result with the given settings and fold parameters."""
    return _BY_NAME[modality].render(raw, settings, params or FoldParams())


def lane_key(modality: str, settings: Settings, params: FoldParams | None = None) -> str:
    """
    Everything a drawn lane depends on besides the recording: the raw result's
    own key, the boundary definition and every rendering parameter. Changing
    the definition therefore never serves a lane drawn for the other one — and
    never costs a model call, because the raw result underneath is unchanged.
    """
    params = params or FoldParams()
    return "|".join(str(v) for v in (
        _CURVE_FORMAT, _BY_NAME[modality].raw_key(settings),
        settings.boundary_definition, settings.spread_s,
        settings.event_channels, params.audio_anchor,
        params.peak_offset_s, params.video_theta, params.video_min_rest_s,
    ))


def extract_raws(
    session: RecordingSession,
    on_result: Callable[[str, Any], None],
    on_error: Callable[[str, str], None],
    settings: Settings | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    use_cache: bool = True,
    modalities: Sequence[str] | None = None,
) -> None:
    """
    The raw result of every modality, reported as each finishes.

    Runs on the caller's thread; one modality failing does not stop the others.

    :param on_result: called as (modality, raw) when a modality finishes.
    :param on_error: called as (modality, message) when a modality raises.
    :param on_progress: called as (modality, done, total) during long modalities.
    :param is_cancelled: polled between and inside modalities; True stops early.
    :param use_cache: reuse and store what is cached in the session directory.
    :param modalities: extract only these; all when None.
    """
    settings = settings or Settings()
    for extractor in _EXTRACTORS:
        if is_cancelled is not None and is_cancelled():
            return
        if modalities is not None and extractor.MODALITY not in modalities:
            continue
        modality = extractor.MODALITY
        try:
            raw = extractor.extract_raw(
                session,
                settings,
                use_cache=use_cache,
                on_progress=(
                    None if on_progress is None
                    else lambda done, total, m=modality: on_progress(m, done, total)
                ),
                is_cancelled=is_cancelled,
            )
        except Exception as exc:                  # noqa: BLE001 — reported, not hidden
            on_error(modality, str(exc))
            continue
        # Outside the try on purpose: a caller that fails to take the result did
        # not fail at segmenting, and reporting it as a modality error hides both.
        on_result(modality, raw)


def segment(
    session: RecordingSession,
    on_result: Callable[[str, BoundaryEvidence], None],
    on_error: Callable[[str, str], None],
    settings: Settings,
    on_progress: Callable[[str, int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    use_cache: bool = True,
) -> None:
    """
    Segment one recording for display, reporting each modality as it finishes.

    The lanes are drawn with the default fold parameters — the feature dialog
    shows a recording, not a fold. A lane already stored under the same key is
    loaded; otherwise it is drawn from the raw result and stored.

    :param settings: with the τ the lanes are drawn with.
    :param use_cache: reuse the lanes, raw results and model verdicts cached in
        the session directory. False recomputes everything and stores nothing.
    """
    pending = []
    for modality in MODALITIES:
        key = lane_key(modality, settings)
        evidence = store.load(session, modality, key) if use_cache else None
        if evidence is not None:
            on_result(modality, evidence)
        else:
            pending.append(modality)
    if not pending:
        return

    def finished(modality: str, raw: Any) -> None:
        evidence = render(modality, raw, settings)
        if use_cache and not (is_cancelled is not None and is_cancelled()):
            store.save(session, modality, evidence, lane_key(modality, settings))
        on_result(modality, evidence)

    extract_raws(
        session, finished, on_error, settings,
        on_progress=on_progress, is_cancelled=is_cancelled, use_cache=use_cache,
        modalities=pending,
    )
