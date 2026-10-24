"""
Action boundaries from events.json alone. Opens exactly one file, so the
modality stays independent for the ablation.

Two stages, so that calibrating a parameter never re-reads anything (C8):

    extract_raw(session)   the input bursts, each with its class histogram and
                           whether it was completed — fold-independent
    render(raw, ...)       three sub-curves drawn at the burst starts — cheap

The sub-curves (C14), all placed at the FIRST event of a burst because the
annotation marks the beginning of the next step. The primary run uses the
rest alone (MA 3.2.5); the other two are the "all" sensitivity variant:

    rest          how long the user was idle before the burst, relative to
                  the session's longest pause
    completion    the previous burst ended with a completion key (Enter,
                  Ctrl+Enter, Ctrl+S) not followed by an abort key (Esc,
                  Ctrl+Z) — the chunking signal of Rebmann & van der Aa
    class_change  1 − cosine similarity of the class histograms of the
                  previous and this burst — a simplified context switch

UI labels are not recorded, so completion is approximated by key classes and a
click carries no completion information — a stated limit of the modality.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from docupilot.config import (
    DOUBLE_CLICK_S,
    EVENT_CHANNEL_WEIGHTS,
    KEY_CLASSES,
    MODIFIER_KEYS,
    FoldParams,
    Settings,
)
from docupilot.recording.session import RecordingSession
from docupilot.segmentation.evidence import (
    BOUNDARY_THRESHOLD,
    BoundaryEvidence,
    draw_gaussian,
    grid,
)

MODALITY = "events"

CHANNELS: tuple[str, ...] = ("rest", "completion", "class_change")

# Action type × key class. Releases and modifier keys are no action of their own.
EVENT_CLASSES: tuple[str, ...] = (
    "key:character", "key:navigation", "key:completion", "key:abort", "key:shortcut",
    "mouse:click", "mouse:double_click", "mouse:right_click", "mouse:scroll",
)

# The recorder's own lifecycle events that state how long the log ran; taking
# the length from the video would make the last burst depend on another modality.
_START_EVENT = "recording_started"
_END_EVENT_TYPES = frozenset({"recording_stopped", "av_stopped", "input_stopped"})

_KEY_TO_CLASS: dict[str, str] = {
    key: group for group, keys in KEY_CLASSES.items() for key in keys
}
_KEY_TO_MODIFIER: dict[str, str] = {
    key: group for group, keys in MODIFIER_KEYS.items() for key in keys
}
# Held modifiers that turn a key into a shortcut; Shift only changes the
# character, AltGr types one.
_SHORTCUT_MODIFIERS = ("ctrl", "alt", "cmd")


@dataclass(frozen=True)
class Burst:
    """A run of input events without a pause longer than the burst pause."""

    start_s: float
    end_s: float
    classes: dict[str, int]
    completed: bool


@dataclass(frozen=True)
class EventsRaw:
    """Everything the curves are drawn from; no label and no parameter of a fold."""

    bursts: list[Burst]
    log_start_s: float
    log_end_s: float
    unknown_keys: dict[str, int] = field(default_factory=dict)


# ── Classification ────────────────────────────────────────────────────────────

def _key_name(key: str, held: set[str]) -> tuple[str, bool]:
    """
    The key as the class table spells it, and whether a shortcut modifier was
    held. A control character (Windows reports Ctrl+S as '\\x13') is read back
    as its letter with Ctrl.
    """
    prefix = [m for m in _SHORTCUT_MODIFIERS if m in held and "alt_gr" not in held]
    if len(key) == 1 and ord(key) < 32:
        key = chr(ord(key) + 96)
        if "ctrl" not in prefix:
            prefix.insert(0, "ctrl")
    if prefix and len(key) == 1:
        key = key.lower()
    return "+".join([*prefix, key]), bool(prefix)


def classify_key(key: str, held: set[str], unknown: Counter[str]) -> str:
    """The class of one key press; an unlisted key counts as unknown."""
    name, modified = _key_name(key, held)
    if name in _KEY_TO_CLASS:
        return _KEY_TO_CLASS[name]
    if not modified and len(name) == 1 and name.isprintable():
        return "character"
    if not modified:
        unknown[name] += 1
    return "shortcut"


def classify(events: Sequence[dict[str, Any]]) -> tuple[list[tuple[float, str | None]], dict[str, int]]:
    """
    Every input event as (t_s, class), chronologically; the class is None for
    events that are no action of their own (releases, modifier keys).

    :return: the classified events and the count of every unknown key.
    """
    held: set[str] = set()
    unknown: Counter[str] = Counter()
    last_left_click = -np.inf
    out: list[tuple[float, str | None]] = []

    for event in sorted(events, key=lambda e: float(e.get("t_ms", 0.0))):
        t_s = float(event.get("t_ms", 0.0)) / 1000.0
        kind = event.get("type")
        cls: str | None = None
        if kind in ("key_press", "key_release"):
            key = str(event.get("key", ""))
            modifier = _KEY_TO_MODIFIER.get(key)
            if modifier is not None:
                (held.add if kind == "key_press" else held.discard)(modifier)
            elif kind == "key_press":
                cls = "key:" + classify_key(key, held, unknown)
        elif kind == "mouse_click" and event.get("pressed", True):
            button = str(event.get("button", ""))
            if button.endswith("right"):
                cls = "mouse:right_click"
            elif button.endswith("left") and t_s - last_left_click <= DOUBLE_CLICK_S:
                cls = "mouse:double_click"
                last_left_click = -np.inf
            else:
                cls = "mouse:click"
                if button.endswith("left"):
                    last_left_click = t_s
        elif kind == "mouse_scroll":
            cls = "mouse:scroll"
        out.append((t_s, cls))
    return out, dict(unknown)


def split_into_bursts(times_s: Sequence[float], pause_s: float) -> list[tuple[int, int]]:
    """
    Index ranges [first, last] of the bursts: a new burst starts when the gap
    to the previous event is longer than `pause_s` (strictly greater).
    """
    if not len(times_s):
        return []
    out: list[tuple[int, int]] = []
    first = 0
    for i in range(1, len(times_s)):
        if times_s[i] - times_s[i - 1] > pause_s:
            out.append((first, i - 1))
            first = i
    out.append((first, len(times_s) - 1))
    return out


def _completed(classes_in_order: Sequence[str]) -> bool:
    """A completion key, and no abort key after the last one."""
    last_completion = max(
        (i for i, c in enumerate(classes_in_order) if c == "key:completion"), default=-1
    )
    if last_completion < 0:
        return False
    return "key:abort" not in classes_in_order[last_completion + 1:]


def bursts_from_events(events: Sequence[dict[str, Any]], pause_s: float) -> tuple[list[Burst], dict[str, int]]:
    """Group input events into bursts with their class histogram and completion."""
    classified, unknown = classify(events)
    times = [t for t, _ in classified]
    bursts = []
    for first, last in split_into_bursts(times, pause_s):
        classes = [c for _, c in classified[first:last + 1] if c is not None]
        bursts.append(Burst(
            start_s=times[first],
            end_s=times[last],
            classes=dict(Counter(classes)),
            completed=_completed(classes),
        ))
    return bursts, unknown


# ── Stage 1: raw ──────────────────────────────────────────────────────────────

def raw_key(settings: Settings) -> str:
    """Everything a raw result depends on besides the recording."""
    return f"burst_pause={settings.burst_pause_s}|double_click={DOUBLE_CLICK_S}"


def _log_bounds_s(events: Sequence[dict[str, Any]]) -> tuple[float, float]:
    """
    When the log started and stopped, by its own lifecycle events. Without a
    start event the clock origin is used; without a stop event the last event.
    """
    if not events:
        return 0.0, 0.0
    starts = [float(e.get("t_ms", 0.0)) for e in events if e.get("type") == _START_EVENT]
    ends = [float(e.get("t_ms", 0.0)) for e in events if e.get("type") in _END_EVENT_TYPES]
    start = min(starts) / 1000.0 if starts else 0.0
    end = max(ends or [float(e.get("t_ms", 0.0)) for e in events]) / 1000.0
    return max(start, 0.0), end


def extract_raw(
    session: RecordingSession,
    settings: Settings | None = None,
    *,
    use_cache: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> EventsRaw:
    """
    The bursts of the event log. Parsing one JSON file is cheaper than any
    cache, so `use_cache` and the callbacks — the shared extractor contract —
    are unused here.

    :param settings: supplies the burst pause (a priori 2.0 s, C12 sensitivity).
    """
    settings = settings or Settings()
    log = session.read_events()
    log_start, log_end = _log_bounds_s(log)
    bursts, unknown = bursts_from_events(session.input_events_of(log), settings.burst_pause_s)
    return EventsRaw(bursts=bursts, log_start_s=log_start, log_end_s=log_end,
                     unknown_keys=unknown)


# ── Stage 2: render ───────────────────────────────────────────────────────────

def _cosine(a: dict[str, int], b: dict[str, int]) -> float | None:
    va = np.array([a.get(c, 0) for c in EVENT_CLASSES], dtype=np.float64)
    vb = np.array([b.get(c, 0) for c in EVENT_CLASSES], dtype=np.float64)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return None
    return float(va @ vb / (na * nb))


def transitions(
    raw: EventsRaw, definition: str = "begin"
) -> list[tuple[float, dict[str, float]]]:
    """
    One (position, sub-curve heights) per burst transition.

    "begin": at the first event of burst k, describing the step from k−1 to k;
    the first burst's rest is measured from the start of the log.
    "end" (sensitivity only): at the last event of burst k, describing the step
    from k to k+1; the last burst's rest runs to the end of the log.

    The rest height follows the duration of the pause (MA 3.2.5): the pause
    divided by the session's longest one, so it lies in (0, 1], grows with
    the pause and is normalised to the user's own pace without a parameter.
    """
    bursts = raw.bursts
    out: list[tuple[float, dict[str, float]]] = []
    rests: list[float] = []
    for k, burst in enumerate(bursts):
        if definition == "begin":
            before = bursts[k - 1] if k > 0 else None
            after = burst
            position = burst.start_s
            rest = burst.start_s - (before.end_s if before else raw.log_start_s)
        else:
            before = burst
            after = bursts[k + 1] if k + 1 < len(bursts) else None
            position = burst.end_s
            rest = (after.start_s if after else raw.log_end_s) - burst.end_s
        similarity = (
            _cosine(before.classes, after.classes) if before is not None and after is not None
            else None
        )
        rests.append(rest)
        out.append((position, {
            "completion": 1.0 if before is not None and before.completed else 0.0,
            "class_change": 0.0 if similarity is None else 1.0 - similarity,
        }))
    longest = max(rests, default=0.0)
    heights_rest = [max(r, 0.0) / longest if longest > 0 else 0.0 for r in rests]
    return [(position, {"rest": float(r), **heights})
            for (position, heights), r in zip(out, heights_rest)]


def channel_names(settings: Settings) -> tuple[str, ...]:
    """The sub-curves a run uses: all three, or only the rest (C14 sensitivity)."""
    return CHANNELS if settings.event_channels == "all" else ("rest",)


def render(
    raw: EventsRaw, settings: Settings, params: FoldParams | None = None
) -> BoundaryEvidence:
    """
    Draw the bursts onto the sub-curves: per transition and sub-curve a bell
    at the transition's time, height = the sub-curve's value, as wide as the
    tolerance window (σ = τ/3, cut off at ±τ).

    The displayed score — and what the rule decider reads — is the weighted
    maximum of the sub-curves (1 : 1 : 0.5, a priori). A boundary is committed
    where it reaches BOUNDARY_THRESHOLD.

    :param params: unused — the event curves have no in-fold parameter.
    """
    times_s = grid(raw.log_end_s)
    if not raw.bursts or len(times_s) == 0:
        return BoundaryEvidence.empty()

    names = channel_names(settings)
    curves = {name: np.zeros(len(times_s), dtype=np.float32) for name in names}

    boundaries_s: list[float] = []
    for position, heights in transitions(raw, settings.boundary_definition):
        for name in names:
            draw_gaussian(curves[name], times_s, position, heights[name], settings.spread_s)
        combined = max(EVENT_CHANNEL_WEIGHTS[n] * heights[n] for n in names)
        if combined >= BOUNDARY_THRESHOLD:
            boundaries_s.append(float(position))

    score = np.max([EVENT_CHANNEL_WEIGHTS[n] * curves[n] for n in names], axis=0)
    return BoundaryEvidence(times_s, score.astype(np.float32), boundaries_s, channels=curves)
