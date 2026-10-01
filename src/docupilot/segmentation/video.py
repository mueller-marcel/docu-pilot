"""
Action boundaries from the screen recording alone: pHash tile activity finds the
frames where the screen stands still, and a VLM judges each settled state against
an ANCHOR holding the last established one.

Reads only the video stream, so the modality stays independent for the ablation.

Two stages (C8):

    extract_raw(session)   activity scan + one verdict per judged dwell —
                           expensive, cached, independent of the definition
    render(raw, settings)  where each verdict is drawn — cheap

The module is layered so the decisions are separable from the decoding: the
activity signal is a function of a frame SEQUENCE, the dwell segmentation a
function of that signal, and the anchor walk a function of the dwells plus two
callables. Only `_scan`, `_read_frames` and `_ffprobe_times_s` touch OpenCV and
ffprobe.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from docupilot.config import FoldParams, Settings
from docupilot.recording.session import RecordingSession
from docupilot.segmentation import store
from docupilot.segmentation.evidence import (
    BOUNDARY_THRESHOLD,
    BoundaryEvidence,
    apply_gaussian,
)
from docupilot.segmentation.video_scoring import Judgement

MODALITY = "video"

_PHASH_SIZE = 8
_ACTIVITY_GRID = 8

# FROZEN (C8). They decide the dwells and therefore the VLM calls, so tuning
# them in-fold is not affordable. Swept once on a development session for the
# cheapest setting at full recall; the values are frozen and their origin
# documented, a sensitivity run exists only on a sub-corpus.
ACTIVITY_QUIET = Settings().activity_quiet   # per-tile pHash distance below which a frame is "still"
_MIN_DWELL_S = 0.5                           # a still run this long is a settled state

# SET, not tuned. How far INTO a dwell the frame shown to the model is sampled;
# the first still frame can still show a fading animation.
_SETTLE_OFFSET_S = 0.2

# SET, not tuned. Mean grey-level difference above which a tile counts as
# changed, for the Set-of-Mark box only — the box only has to point.
_PIXEL_CHANGE_EPS = 1.0

# TECHNICAL. Budget cap: a flickering recording could fragment into thousands of
# dwells. Whether it was hit is reported per session.
_MAX_CALLS = 400

# The activity scan is the expensive, model-free half of this modality and is
# also what the synchronisation report reads. It is a pure function of the
# recording bytes and these two constants, so it is kept beside the recording.
ACTIVITY_FILE = "video_activity.npz"
_ACTIVITY_FORMAT = 1

# What the VLM said about each frame pair, kept beside the recording.
VERDICT_CACHE_FILE = "gui_vlm_cache.json"

Region = tuple[float, float, float, float]


@dataclass(frozen=True)
class ActivityScan:
    """Per-frame stillness of one recording, plus what is needed to place it in time."""

    n_frames: int
    activity: np.ndarray     # (n_frames,) float32 — largest per-tile pHash distance to the previous frame
    fps: float
    times_s: np.ndarray      # (n_frames,) float64 — capture time of each frame on the event clock
    pts_s: np.ndarray        # (n_frames,) float64 — the container's presentation times, first = 0


@dataclass(frozen=True)
class DwellVerdict:
    """What the model said about one dwell, and where that dwell lies."""

    dwell_start: int         # frame index the screen settled at
    dwell_end: int           # last still frame of the dwell
    judgement: Judgement


@dataclass(frozen=True)
class VideoRaw:
    """Everything the curve is drawn from; nothing here depends on a definition."""

    times_s: np.ndarray
    activity: np.ndarray
    fps: float
    quiet: float
    verdicts: list[DwellVerdict]
    calls: int
    limit_hit: bool


# ── Activity signal ───────────────────────────────────────────────────────────

def _phash(gray: np.ndarray):
    import imagehash
    from PIL import Image

    return imagehash.phash(Image.fromarray(gray), hash_size=_PHASH_SIZE)


def _tile_bounds(h: int, w: int) -> list[tuple[int, int, int, int]]:
    """(row0, row1, col0, col1) of every cell of the activity grid, row-major."""
    g = _ACTIVITY_GRID
    return [
        (r * h // g, (r + 1) * h // g, c * w // g, (c + 1) * w // g)
        for r in range(g)
        for c in range(g)
    ]


def _tiles(gray: np.ndarray) -> tuple:
    """One pHash per cell of the activity grid."""
    h, w = gray.shape[:2]
    return tuple(_phash(gray[r0:r1, c0:c1]) for r0, r1, c0, c1 in _tile_bounds(h, w))


def _distance(a: tuple, b: tuple) -> float:
    """Largest per-tile pHash distance, normalised to [0, 1]."""
    return max(float(x - y) for x, y in zip(a, b)) / (_PHASH_SIZE ** 2)


def frame_activity(gray_frames: Iterable[np.ndarray]) -> np.ndarray:
    """
    The activity signal of a sequence of GREYSCALE frames: per frame, the largest
    per-tile pHash distance to its predecessor, normalised to [0, 1].

    Tiles rather than the whole frame because a meaningful change often covers a
    small region (one cell recoloured, an arrow appearing in a header) and would
    average away; the MAXIMUM over tiles rather than the mean for the same
    reason. The first frame has no predecessor and scores 0.

    :param gray_frames: the frames, in order.
    :return: (n_frames,) float32.
    """
    activity: list[float] = []
    previous: tuple | None = None
    for gray in gray_frames:
        current = _tiles(gray)
        activity.append(0.0 if previous is None else _distance(previous, current))
        previous = current
    return np.asarray(activity, dtype=np.float32)


def dwells(
    activity: np.ndarray, min_frames: int, quiet: float = ACTIVITY_QUIET
) -> list[tuple[int, int]]:
    """
    Maximal runs of still frames lasting at least `min_frames`, inclusive bounds.

    The reading direction is inverted on purpose: what is searched is the REST,
    not the change. On a screen recording the surface stands still between two
    operations, and that stillness is more robustly measured than the transition
    that ends it. The minimum duration is the first guard against
    over-segmentation — it discards rests too short to be a settled state.
    """
    out: list[tuple[int, int]] = []
    i, n = 0, len(activity)
    while i < n:
        if activity[i] < quiet:
            j = i
            while j < n and activity[j] < quiet:
                j += 1
            if j - i >= min_frames:
                out.append((i, j - 1))
            i = j
        else:
            i += 1
    return out


def settled_frames(dwell_list: Sequence[tuple[int, int]], fps: float) -> list[int]:
    """The frame shown to the model per dwell: `_SETTLE_OFFSET_S` into it, so a
    fading animation in the first still frame is not what gets judged."""
    offset = max(0, int(_SETTLE_OFFSET_S * fps))
    return [min(end, start + offset) for start, end in dwell_list]


def first_active_frame_after(activity: np.ndarray, frame: int, quiet: float) -> int | None:
    """The first frame after `frame` whose activity reaches `quiet`; None when
    the screen never moves again."""
    later = np.flatnonzero(activity[frame + 1:] >= quiet)
    return None if len(later) == 0 else int(frame + 1 + later[0])


# ── Changed region (the Set-of-Mark hint) ─────────────────────────────────────

def _gray16(bgr: np.ndarray) -> np.ndarray:
    """Grey levels as int16, so a difference cannot wrap around like uint8 would."""
    import cv2

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)


def changed_region(a: np.ndarray, b: np.ndarray) -> Region | None:
    """
    Bounding box (x0, y0, x1, y1 in [0,1]) of the tiles whose PIXELS differ
    between two greyscale frames — the hint that tells the model where to look.

    Pixels, not pHash: on a low-texture tile the hash flips on codec noise alone.

    :return: None when no tile differs measurably, which means the two states are
        effectively identical and there is nothing to pay a model call for.
    """
    g = _ACTIVITY_GRID
    h, w = a.shape[:2]
    rows: list[int] = []
    cols: list[int] = []
    for index, (r0, r1, c0, c1) in enumerate(_tile_bounds(h, w)):
        if float(np.abs(a[r0:r1, c0:c1] - b[r0:r1, c0:c1]).mean()) > _PIXEL_CHANGE_EPS:
            rows.append(index // g)
            cols.append(index % g)
    if not rows:
        return None
    return (min(cols) / g, min(rows) / g, (max(cols) + 1) / g, (max(rows) + 1) / g)


# ── The anchor walk: the modality's actual decision logic ─────────────────────

def walk_dwells(
    steps: Sequence[tuple[int, int, int]],
    has_change: Callable[[int, int], bool],
    judge: Callable[[int, int], Judgement | None],
    *,
    threshold: float = BOUNDARY_THRESHOLD,
    max_calls: int = _MAX_CALLS,
    on_call: Callable[[int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    on_limit: Callable[[], None] | None = None,
) -> list[DwellVerdict]:
    """
    Judge each settled state against an ANCHOR holding the last ESTABLISHED one.

    This is the core of the video modality. Comparing dwell *i* against dwell
    *i−1* would make every judgement local: opening a menu, stepping into a
    submenu and finally clicking the entry that changes something would produce
    three unrelated comparisons. Against an anchor, the intermediate states are
    rejected AND leave the anchor where it was, so the final question is "has
    anything been completed since the state the user started from?" — the
    question that matters.

    Setting a boundary and advancing the anchor are therefore ONE decision.

    :param steps: (dwell_start, dwell_end, settled_frame) per dwell, in order.
        The first settled frame becomes the initial anchor and is never judged.
    :param has_change: (anchor, current) -> is there anything to compare at all.
        A False costs no model call, which is why it is separate from `judge`.
    :param judge: (anchor, current) -> the verdict, or None when the model gave
        no usable answer; that pair then gets no evidence rather than a guess.
    :param threshold: at or above this p_boundary the anchor advances.
    :param max_calls: budget for `judge` calls; `on_limit` is called when hit.
    :param on_call: called with the running call count after each judged pair.
    :param is_cancelled: polled before each pair; True stops and keeps what was
        gathered so far.
    :return: one entry per JUDGED pair that produced a verdict, in order.
    """
    verdicts: list[DwellVerdict] = []
    anchor: int | None = None
    calls = 0

    for dwell_start, dwell_end, current in steps:
        if anchor is None:                       # the workflow's starting point
            anchor = current
            continue
        if is_cancelled is not None and is_cancelled():
            break
        if calls >= max_calls:
            if on_limit is not None:
                on_limit()
            break
        if not has_change(anchor, current):
            continue

        judgement = judge(anchor, current)
        calls += 1
        if on_call is not None:
            on_call(calls)
        if judgement is None:
            continue

        verdicts.append(DwellVerdict(dwell_start, dwell_end, judgement))
        if judgement.p_boundary >= threshold:
            anchor = current

    return verdicts


def evidence_from_verdicts(
    verdicts: Sequence[DwellVerdict],
    activity: np.ndarray,
    times_s: np.ndarray,
    fps: float,
    *,
    quiet: float = ACTIVITY_QUIET,
    definition: str = "begin",
    spread_s: float = 1.0,
    threshold: float = BOUNDARY_THRESHOLD,
) -> BoundaryEvidence:
    """
    Draw the verdicts onto the score curve.

    What is judged is unchanged: the settled state of a dwell against the
    anchor, "has an operation been completed since the last established
    state?". Only WHERE the answer is drawn depends on the definition:

    "begin": at the first frame after the dwell that moves again — the next
        visible activity, where the next step begins. A dwell after which the
        screen never moves again has no next step and gets no evidence.
    "end" (sensitivity only): at the dwell's first frame, the instant the
        result settled.

    The cursor is not in the video, so under "begin" the evidence arrives with
    the first SCREEN change and may trail an annotation that saw a cursor move
    or a click. That lag is deliberately not calibrated away — the window
    features absorb it.
    """
    score = np.zeros(len(times_s), dtype=np.float32)
    spread = max(1, int(spread_s * fps))
    boundaries_s: list[float] = []
    for verdict in verdicts:
        if definition == "begin":
            frame = first_active_frame_after(activity, verdict.dwell_end, quiet)
            if frame is None:
                continue
        else:
            frame = verdict.dwell_start
        p = verdict.judgement.p_boundary
        apply_gaussian(score, frame, p, spread)
        if p >= threshold:
            boundaries_s.append(float(times_s[frame]))
    return BoundaryEvidence(times_s, score, boundaries_s)


def render(
    raw: VideoRaw, settings: Settings | None = None, params: FoldParams | None = None
) -> BoundaryEvidence:
    """The curve of a raw result; the video has no in-fold parameter."""
    settings = settings or Settings()
    return evidence_from_verdicts(
        raw.verdicts, raw.activity, raw.times_s, raw.fps,
        quiet=raw.quiet,
        definition=settings.boundary_definition,
        spread_s=settings.spread_s,
    )


# ── Decoding (the only part that touches OpenCV and ffprobe) ──────────────────

def parse_frame_times(csv_text: str, n_frames: int) -> np.ndarray:
    """
    Frame presentation times from ffprobe's CSV output, zeroed on the first frame.

    :raises RuntimeError: when the count does not match the decoded frames — a
        mismatch means the container and the decoder disagree, and silently
        trusting either would put the whole modality on a wrong time axis.
    """
    times = [float(line.split(",")[0]) for line in csv_text.splitlines() if line.strip()]
    if len(times) != n_frames:
        raise RuntimeError(
            f"ffprobe lieferte {len(times)} Frame-Zeiten, dekodiert wurden "
            f"{n_frames} — inkonsistente Aufnahme."
        )
    t = np.asarray(times, dtype=np.float64)
    return t - t[0] if len(t) else t


def _ffprobe_times_s(video_path: str, n_frames: int) -> np.ndarray:
    """The container's presentation time of each frame, read straight from the MP4."""
    import subprocess

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pts_time", "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    ).stdout
    return parse_frame_times(out, n_frames)


def capture_times_s(session: RecordingSession, n_frames: int, pts_s: np.ndarray) -> np.ndarray:
    """
    The capture time of each frame on the clock the events use.

    Recordings with frame_times.npy carry it directly — the recorder stamped
    each frame with the monotonic clock before piping it (C9). Older recordings
    fall back to the container's presentation times, which ffmpeg took from its
    wall clock and which share the events' origin only to within the pipe
    start-up latency.

    :raises RuntimeError: when frame_times.npy does not hold one time per
        decoded frame — ffmpeg dropped or duplicated frames and no frame can be
        placed reliably.
    """
    path = session.frame_times_path
    if not path.exists():
        return pts_s
    times = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if len(times) != n_frames:
        raise RuntimeError(
            f"frame_times.npy enthält {len(times)} Zeiten, dekodiert wurden {n_frames} "
            "Frames — ffmpeg hat Frames verworfen oder dupliziert."
        )
    return times


def _scan(video_path: str) -> tuple[int, np.ndarray, float]:
    """Stream the video once; return frame count, per-frame activity and the fps."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, np.zeros(0, dtype=np.float32), 25.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    def frames():
        try:
            while True:
                ret, bgr = cap.read()
                if not ret:
                    return
                yield cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        finally:
            cap.release()

    activity = frame_activity(frames())
    return len(activity), activity, fps


def _read_frames(video_path: str, wanted: set[int]) -> dict[int, np.ndarray]:
    """
    Fetch the given frames in one linear pass, downscaled on read.

    Sequential decoding beats seeking on a long-GOP MP4, and full-size frames
    would hold ~0.5 GB in memory for nothing — the model only sees the downscale.
    """
    frames: dict[int, np.ndarray] = {}
    if not wanted:
        return frames                      # nothing to decode, nothing to import

    import cv2

    from docupilot.segmentation.video_scoring import downscale

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return frames
    last = max(wanted)
    try:
        idx = 0
        while idx <= last:
            ret, bgr = cap.read()
            if not ret:
                break
            if idx in wanted:
                frames[idx] = downscale(bgr)
            idx += 1
    finally:
        cap.release()
    return frames


# What a stored scan depends on besides the recording, which is never edited: a
# scan taken on another grid would be read against the wrong threshold.
_ACTIVITY_KEY = f"{_ACTIVITY_FORMAT}|{_PHASH_SIZE}|{_ACTIVITY_GRID}"


def scan_activity(session: RecordingSession, use_cache: bool = True) -> ActivityScan:
    """
    The per-frame activity of a recording, read from the session directory when
    it was scanned before, decoded otherwise.

    :param session: the recording; its .recording_path points at the MP4.
    :param use_cache: read a stored scan and store a fresh one. False decodes
        and stores nothing.
    :raises RuntimeError: when the frame times cannot be read (see
        parse_frame_times and capture_times_s).
    """
    path = session.session_dir / ACTIVITY_FILE
    key = _ACTIVITY_KEY if use_cache else None

    loaded: tuple[int, np.ndarray, float, np.ndarray] | None = None
    if key is not None and path.exists():
        try:
            with np.load(path, allow_pickle=False) as data:
                if str(data["key"]) == key:
                    loaded = (
                        int(data["n_frames"]),
                        np.asarray(data["activity"], dtype=np.float32),
                        float(data["fps"]),
                        np.asarray(data["times_s"], dtype=np.float64),
                    )
        except Exception:                # noqa: BLE001 — a stale or damaged file is a miss
            loaded = None

    if loaded is None:
        video_path = str(session.recording_path)
        n_frames, activity, fps = _scan(video_path)
        pts_s = _ffprobe_times_s(video_path, n_frames)
        if key is not None:
            store.write_npz(path, key=np.array(key), n_frames=np.array(n_frames),
                            activity=activity, fps=np.array(fps), times_s=pts_s)
    else:
        n_frames, activity, fps, pts_s = loaded

    return ActivityScan(
        n_frames=n_frames, activity=activity, fps=fps,
        times_s=capture_times_s(session, n_frames, pts_s), pts_s=pts_s,
    )


# ── Stage 1: raw ──────────────────────────────────────────────────────────────

def raw_key(settings: Settings) -> str:
    """Everything a raw result depends on besides the recording."""
    from docupilot.segmentation import video_scoring as vlm

    return "|".join(str(v) for v in (
        _ACTIVITY_KEY, settings.activity_quiet, _MIN_DWELL_S, _SETTLE_OFFSET_S,
        _PIXEL_CHANGE_EPS, _MAX_CALLS, vlm.MODEL, vlm.PROMPT_VERSION,
    ))


def _to_payload(verdicts: Sequence[DwellVerdict], calls: int, limit_hit: bool) -> dict:
    return {
        "calls": calls,
        "limit_hit": limit_hit,
        "verdicts": [
            {"start": v.dwell_start, "end": v.dwell_end, "category": v.judgement.category,
             "p": v.judgement.p_boundary, "reason": v.judgement.reason,
             "model": v.judgement.model, "retrieved_at": v.judgement.retrieved_at}
            for v in verdicts
        ],
    }


def _from_payload(payload: dict) -> tuple[list[DwellVerdict], int, bool]:
    verdicts = [
        DwellVerdict(int(v["start"]), int(v["end"]), Judgement(
            category=str(v["category"]), p_boundary=float(v["p"]),
            reason=str(v.get("reason", "")), model=str(v.get("model", "")),
            retrieved_at=str(v.get("retrieved_at", "")),
        ))
        for v in payload["verdicts"]
    ]
    return verdicts, int(payload["calls"]), bool(payload["limit_hit"])


def extract_raw(
    session: RecordingSession,
    settings: Settings | None = None,
    *,
    use_cache: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    reuse_verdicts: bool = True,
) -> VideoRaw:
    """
    Dwells -> a VLM judges each against the anchor -> the anchor advances only
    on an accepted boundary.

    Runs for minutes on a miss; must not be called on a UI thread.

    :param settings: supplies ACTIVITY_QUIET (frozen; a variant re-extracts).
    :param use_cache: reuse the raw result, the verdicts and the activity scan
        cached in the session directory, and store fresh ones.
    :param on_progress: called as (judged_pairs, total_pairs) after each pair.
    :param is_cancelled: polled before each pair; True stops early and returns
        what was gathered so far, which is then never stored.
    :param reuse_verdicts: False asks the model again for every pair while still
        reading the activity scan — what the repeatability check needs.
    :raises RuntimeError: when no VLM backend is reachable.
    """
    from docupilot.segmentation import video_scoring as vlm

    settings = settings or Settings()
    quiet = settings.activity_quiet
    scan = scan_activity(session, use_cache=use_cache)
    key = raw_key(settings)

    def raw(verdicts: list[DwellVerdict], calls: int, limit_hit: bool) -> VideoRaw:
        return VideoRaw(scan.times_s, scan.activity, scan.fps, quiet, verdicts, calls, limit_hit)

    cached_raw = use_cache and reuse_verdicts
    if cached_raw:
        payload = store.load_raw(session, MODALITY, key)
        if payload is not None:
            try:
                return raw(*_from_payload(payload))
            except (KeyError, TypeError, ValueError):
                pass                              # an unreadable result is a miss

    if not vlm.is_available():
        raise RuntimeError(
            f"Cloud-Modell '{vlm.MODEL}' nicht nutzbar. Benötigt:\n"
            "  poetry install         (Paket 'anthropic')\n"
            "  ANTHROPIC_API_KEY=...  oder  ant auth login\n\n"
            "Ohne VLM kann die Video-Modalität keine Handlungsgrenzen bestimmen — "
            "ein rein struktureller Score misst Pixelmenge, nicht Bedeutung."
        )

    dwell_list = (
        dwells(scan.activity, max(1, round(_MIN_DWELL_S * scan.fps)), quiet)
        if scan.n_frames >= 2 else []
    )
    if len(dwell_list) < 2:
        result = raw([], 0, False)
        if cached_raw:
            store.save_raw(session, MODALITY, key, _to_payload([], 0, False))
        return result

    settled = settled_frames(dwell_list, scan.fps)
    halves = _read_frames(str(session.recording_path), set(settled))
    steps = [
        (start, end, frame)
        for (start, end), frame in zip(dwell_list, settled) if frame in halves
    ]

    # The anchor is compared against every following dwell, so its grey version
    # is computed once and kept, not rebuilt per comparison.
    grays: dict[int, np.ndarray] = {}

    def gray(index: int) -> np.ndarray:
        if index not in grays:
            grays[index] = _gray16(halves[index])
        return grays[index]

    regions: dict[tuple[int, int], Region] = {}

    def has_change(anchor: int, current: int) -> bool:
        region = changed_region(gray(anchor), gray(current))
        if region is None:
            return False
        regions[(anchor, current)] = region
        return True

    cache = (
        vlm.Cache(session.session_dir / VERDICT_CACHE_FILE)
        if use_cache and reuse_verdicts else None
    )
    calls = 0
    limit_hit = False

    def judge(anchor: int, current: int) -> Judgement | None:
        nonlocal calls
        calls += 1
        composite = vlm.encode_pair(
            halves[anchor], halves[current], regions[(anchor, current)]
        )
        return vlm.judge(composite, cache=cache)

    def limit_reached() -> None:
        # A silent cap reads as "everything covered". Say what was dropped so
        # the missing tail is visible, not mistaken for "no more boundaries".
        nonlocal limit_hit
        limit_hit = True
        warnings.warn(
            f"Video: _MAX_CALLS={_MAX_CALLS} erreicht in {session.session_dir.name} "
            f"— restliche Dwells ungeprüft, spätere Grenzen fehlen.",
            stacklevel=2,
        )

    total = max(1, len(steps) - 1)
    verdicts = walk_dwells(
        steps, has_change, judge,
        on_call=(None if on_progress is None else lambda done: on_progress(done, total)),
        is_cancelled=is_cancelled,
        on_limit=limit_reached,
    )

    if cache is not None:
        cache.flush()                # also on cancel: keep what we paid for
    cancelled = is_cancelled is not None and is_cancelled()
    if cached_raw and not cancelled:
        store.save_raw(session, MODALITY, key, _to_payload(verdicts, calls, limit_hit))
    return raw(verdicts, calls, limit_hit)
