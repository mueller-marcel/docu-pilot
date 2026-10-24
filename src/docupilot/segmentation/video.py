"""
Action boundaries from the screen recording alone (implementation guide to
section 3.2.3): the SSIM of every frame to its predecessor finds the rests,
every change between two rests is a candidate, and a VLM judges from the
BEFORE|AFTER pair whether that change is the first step toward a NEW sub-goal.
The boundary lies where that change begins — the first frame that moves again,
the first visible hint of the new sub-goal (MA, 3.1.3).

Reads only the video stream, so the modality stays independent for the ablation.

Two stages:

    extract_raw(session)        the SSIM scan — model-free, cached, independent
                                of every fold
    render(raw, settings, p)    the changes under the fold's noise threshold θ and
                                minimum rest d, one verdict per change (cached per
                                pair), and the bells drawn at their onsets

θ and d are chosen per fold from the training sessions (calibration.py); only
the changes they produce cost model calls.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from docupilot.config import FoldParams, Settings
from docupilot.recording.session import RecordingSession
from docupilot.segmentation import store
from docupilot.segmentation.evidence import (
    BOUNDARY_THRESHOLD,
    BoundaryEvidence,
    draw_gaussian,
    grid,
)
from docupilot.segmentation.video_scoring import PANEL_WIDTH, Judgement, downscale

MODALITY = "video"

_ACTIVITY_GRID = 8

# SSIM after Wang et al. (2004): Gaussian window σ = 1.5 (11 × 11),
# C1 = (0.01 L)², C2 = (0.03 L)² for L = 255 grey levels.
_SSIM_SIGMA = 1.5
_SSIM_C1 = (0.01 * 255) ** 2
_SSIM_C2 = (0.03 * 255) ** 2

# Set, not tuned: the first still frame of a rest can still show a fading
# animation.
_SETTLE_OFFSET_S = 0.2

# Set, not tuned.
_PIXEL_CHANGE_EPS = 1.0

_MAX_CALLS = 400

SSIM_FILE = "video_ssim.npz"
_SSIM_FORMAT = 1

VERDICT_CACHE_FILE = "gui_vlm_cache.json"

Region = tuple[float, float, float, float]


@dataclass(frozen=True)
class Change:
    """The screen moving from one rest to the next."""

    rest_start: int
    onset: int
    before: int
    after: int


@dataclass(frozen=True)
class ChangeVerdict:
    """What the model said about one change, and where that change lies."""

    rest_start: int
    onset: int
    judgement: Judgement


@dataclass(frozen=True)
class Judged:
    """The verdicts of one session under one (θ, d)."""

    verdicts: list[ChangeVerdict]
    n_changes: int
    calls: int
    limit_hit: bool


@dataclass(eq=False)
class VideoRaw:
    """
    The SSIM scan of one recording, and what the VLM said about its changes
    under each (θ, d) a fold asked for — judged on first use, then kept.
    """

    recording_path: Path
    session_dir: Path
    times_s: np.ndarray
    sim: np.ndarray
    reuse_verdicts: bool = True
    on_progress: Callable[[int, int], None] | None = None
    is_cancelled: Callable[[], bool] | None = None
    judged: dict[tuple[float, float], Judged] = field(default_factory=dict)


def _gray(bgr: np.ndarray) -> np.ndarray:
    """A frame as it is measured: at the width the model sees, grey, float32."""
    import cv2

    return cv2.cvtColor(downscale(bgr), cv2.COLOR_BGR2GRAY).astype(np.float32)


def _tile_bounds(h: int, w: int) -> list[tuple[int, int, int, int]]:
    """(row0, row1, col0, col1) of every cell of the tile grid, row-major."""
    g = _ACTIVITY_GRID
    return [
        (r * h // g, (r + 1) * h // g, c * w // g, (c + 1) * w // g)
        for r in range(g)
        for c in range(g)
    ]


def _blur(x: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.GaussianBlur(x, (11, 11), _SSIM_SIGMA)


def _moments(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(frame, local mean, local mean of the square) — what SSIM needs of ONE
    frame, so each frame is blurred once and not again as the next predecessor."""
    return gray, _blur(gray), _blur(gray * gray)


def _tile_ssim(a: tuple, b: tuple) -> np.ndarray:
    """Mean SSIM per tile, row-major, between two frames given as `_moments`."""
    (x, mx, mxx), (y, my, myy) = a, b
    cov = _blur(x * y) - mx * my
    var_x, var_y = mxx - mx * mx, myy - my * my
    ssim = ((2 * mx * my + _SSIM_C1) * (2 * cov + _SSIM_C2)) / (
        (mx * mx + my * my + _SSIM_C1) * (var_x + var_y + _SSIM_C2))
    h, w = ssim.shape
    return np.array([float(ssim[r0:r1, c0:c1].mean()) for r0, r1, c0, c1 in _tile_bounds(h, w)],
                    dtype=np.float32)


def frame_similarity(gray_frames: Iterable[np.ndarray]) -> np.ndarray:
    """
    s_i = SSIM(frame i, frame i−1) on the luminance, the SMALLEST over the tiles
    so a small change does not vanish in the mean of the whole screen. SSIM
    compares local luminance, contrast and structure, so codec noise barely
    moves it while a new line of text does. s_0 = 1: the first frame has no
    predecessor.

    :param gray_frames: greyscale float frames, in order.
    :return: (n,) float32.
    """
    sim: list[float] = []
    previous: tuple | None = None
    for gray in gray_frames:
        current = _moments(gray)
        sim.append(1.0 if previous is None else float(_tile_ssim(previous, current).min()))
        previous = current
    return np.asarray(sim, dtype=np.float32)


def rest_phases(
    times_s: np.ndarray, sim: np.ndarray, theta: float, min_rest_s: float
) -> list[tuple[int, int]]:
    """
    Frame i is still when s_i ≥ θ. A maximal run of still frames k+1 … j forms,
    together with frame k it follows, a rest when t_j − t_k ≥ d.

    :return: (k, j) per rest — its first and last frame, in order.
    """
    still = np.r_[False, sim[1:] >= theta]
    phases: list[tuple[int, int]] = []
    i, n = 1, len(times_s)
    while i < n:
        if still[i]:
            k, j = i - 1, i
            while j + 1 < n and still[j + 1]:
                j += 1
            if times_s[j] - times_s[k] >= min_rest_s:
                phases.append((k, j))
            i = j + 1
        else:
            i += 1
    return phases


def state_frame(times_s: np.ndarray, k: int, j: int) -> int:
    """The frame standing for rest (k, j): `_SETTLE_OFFSET_S` after its start, at
    most its last frame — so no fading animation is judged."""
    return min(int(np.searchsorted(times_s, times_s[k] + _SETTLE_OFFSET_S)), j)


def changes(
    times_s: np.ndarray, sim: np.ndarray, theta: float, min_rest_s: float
) -> list[Change]:
    """
    One change between every two consecutive rests: onset = first moving frame
    after rest m, BEFORE and AFTER = the state frames of rests m and m+1.

    The first change is kept: it begins the first sub-goal, and the annotation
    marks that beginning too (every session has a boundary in its first
    seconds). Movement after the last rest has no AFTER and yields no change.
    """
    ph = rest_phases(times_s, sim, theta, min_rest_s)
    return [
        Change(rest_start=k0, onset=j0 + 1,
               before=state_frame(times_s, k0, j0), after=state_frame(times_s, k1, j1))
        for (k0, j0), (k1, j1) in zip(ph, ph[1:])
    ]


def _gray16(bgr: np.ndarray) -> np.ndarray:
    """Grey levels as int16, so a difference cannot wrap around like uint8 would."""
    import cv2

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)


def changed_region(a: np.ndarray, b: np.ndarray) -> Region | None:
    """
    Bounding box (x0, y0, x1, y1 in [0,1]) of the tiles whose mean grey-level
    difference between two int16 greyscale frames exceeds `_PIXEL_CHANGE_EPS` —
    the hint that tells the model where to look.

    :return: None when no tile differs, which means the screen returned to the
        same state and there is nothing to pay a model call for.
    """
    g = _ACTIVITY_GRID
    h, w = a.shape[:2]
    hit = [
        index for index, (r0, r1, c0, c1) in enumerate(_tile_bounds(h, w))
        if float(np.abs(a[r0:r1, c0:c1] - b[r0:r1, c0:c1]).mean()) > _PIXEL_CHANGE_EPS
    ]
    if not hit:
        return None
    rows = [i // g for i in hit]
    cols = [i % g for i in hit]
    return (min(cols) / g, min(rows) / g, (max(cols) + 1) / g, (max(rows) + 1) / g)


def judge_changes(raw: VideoRaw, theta: float, min_rest_s: float) -> Judged:
    """
    The verdict on every change of `raw` under (θ, d), judged once and kept on
    the raw result. A change whose BEFORE and AFTER do not differ costs no call
    and gets no evidence.

    Runs for minutes on a miss; must not be called on a UI thread. A
    cancelled run returns what it gathered and is not kept.

    :raises RuntimeError: when no VLM backend is reachable.
    """
    from docupilot.segmentation import video_scoring as vlm

    key = (theta, min_rest_s)
    if key in raw.judged:
        return raw.judged[key]

    change_list = changes(raw.times_s, raw.sim, theta, min_rest_s)
    if not change_list:
        raw.judged[key] = Judged([], 0, 0, False)
        return raw.judged[key]
    if not vlm.is_available():
        raise RuntimeError(
            f"Cloud model '{vlm.MODEL}' not usable. Requires:\n"
            "  uv sync                (package 'anthropic')\n"
            "  ANTHROPIC_API_KEY=...  or  ant auth login\n\n"
            "Without a VLM the video modality cannot determine action boundaries — "
            "a purely structural score measures amounts of pixels, not meaning."
        )

    frames = _read_frames(str(raw.recording_path),
                          {c.before for c in change_list} | {c.after for c in change_list})
    cache = vlm.Cache(raw.session_dir / VERDICT_CACHE_FILE) if raw.reuse_verdicts else None
    verdicts: list[ChangeVerdict] = []
    calls = 0
    limit_hit = cancelled = False
    for change in change_list:
        if raw.is_cancelled is not None and raw.is_cancelled():
            cancelled = True
            break
        if change.before not in frames or change.after not in frames:
            continue
        if calls >= _MAX_CALLS:
            # A silent cap reads as "everything covered". Say what was dropped so
            # the missing tail is visible, not mistaken for "no more boundaries".
            limit_hit = True
            warnings.warn(
                f"Video: _MAX_CALLS={_MAX_CALLS} reached in {raw.session_dir.name} "
                f"(θ={theta:g}, d={min_rest_s:g} s) — remaining changes unchecked.",
                stacklevel=2,
            )
            break
        before, after = frames[change.before], frames[change.after]
        region = changed_region(_gray16(before), _gray16(after))
        if region is None:
            continue
        judgement = vlm.judge(vlm.encode_pair(before, after, region), cache=cache)
        calls += 1
        if raw.on_progress is not None:
            raw.on_progress(calls, len(change_list))
        if judgement is not None:
            verdicts.append(ChangeVerdict(change.rest_start, change.onset, judgement))

    if cache is not None:
        cache.flush()                # also on cancel: keep what we paid for
    result = Judged(verdicts, len(change_list), calls, limit_hit)
    if not cancelled:
        raw.judged[key] = result
    return result


def evidence_from_verdicts(
    verdicts: Sequence[ChangeVerdict],
    times_s: np.ndarray,
    *,
    definition: str = "begin",
    sigma_s: float = 1.0,
    threshold: float = BOUNDARY_THRESHOLD,
) -> BoundaryEvidence:
    """
    Draw the verdicts onto the score curve: per verdict a bell at t_q, height
    p_q, σ = `sigma_s` (τ), sampled on the GRID_HZ grid up to the last frame.
    Low p is drawn too, so the curve stays graded.

    "begin": t_q at the change's onset — the first frame that moves again, the
        first visible hint of the new sub-goal (MA, 3.1.3).
    "end" (sensitivity only): at the first frame of the rest before it, the
        instant the previous sub-goal's last result settled.
    """
    if len(times_s) == 0:
        return BoundaryEvidence.empty()
    t = grid(float(times_s[-1]))
    score = np.zeros(len(t), dtype=np.float32)
    boundaries_s: list[float] = []
    for verdict in verdicts:
        frame = verdict.onset if definition == "begin" else verdict.rest_start
        t_q, p_q = float(times_s[frame]), verdict.judgement.p_boundary
        draw_gaussian(score, t, t_q, p_q, sigma_s)
        if p_q >= threshold:
            boundaries_s.append(t_q)
    return BoundaryEvidence(t, score, boundaries_s)


def render(
    raw: VideoRaw, settings: Settings, params: FoldParams | None = None
) -> BoundaryEvidence:
    """The curve under the fold's (θ, d); judges the changes on first use."""
    params = params or FoldParams()
    judged = judge_changes(raw, params.video_theta, params.video_min_rest_s)
    return evidence_from_verdicts(
        judged.verdicts, raw.times_s,
        definition=settings.boundary_definition,
        sigma_s=settings.spread_s,
    )


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
            f"ffprobe returned {len(times)} frame times, but {n_frames} were "
            f"decoded — inconsistent recording."
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


def _scan(video_path: str) -> np.ndarray:
    """Stream the video once; return s_i per frame."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.zeros(0, dtype=np.float32)

    def frames():
        try:
            while True:
                ret, bgr = cap.read()
                if not ret:
                    return
                yield _gray(bgr)
        finally:
            cap.release()

    return frame_similarity(frames())


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


_SSIM_KEY = f"{_SSIM_FORMAT}|{_ACTIVITY_GRID}|{_SSIM_SIGMA}|{PANEL_WIDTH}"


def scan_similarity(session: RecordingSession, use_cache: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    (t_i, s_i) of a recording, read from the session directory when it was
    scanned before, decoded otherwise.

    :param use_cache: read a stored scan and store a fresh one. False decodes
        and stores nothing.
    :raises RuntimeError: when the frame times cannot be read (see
        parse_frame_times).
    """
    path = session.session_dir / SSIM_FILE
    if use_cache and path.exists():
        try:
            with np.load(path, allow_pickle=False) as data:
                if str(data["key"]) == _SSIM_KEY:
                    return (np.asarray(data["times_s"], dtype=np.float64),
                            np.asarray(data["sim"], dtype=np.float32))
        except Exception:                # noqa: BLE001 — a stale or damaged file is a miss
            pass

    video_path = str(session.recording_path)
    sim = _scan(video_path)
    times_s = _ffprobe_times_s(video_path, len(sim))
    if use_cache:
        store.write_npz(path, key=np.array(_SSIM_KEY), times_s=times_s, sim=sim)
    return times_s, sim


def raw_key(settings: Settings) -> str:
    """Everything a raw result and its verdicts depend on besides the recording
    and the fold's (θ, d)."""
    from docupilot.segmentation import video_scoring as vlm

    return "|".join(str(v) for v in (
        _SSIM_KEY, _SETTLE_OFFSET_S, _PIXEL_CHANGE_EPS, _MAX_CALLS,
        vlm.MODEL, vlm.PROMPT_VERSION,
    ))


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
    The SSIM scan of a recording — no model call. The verdicts follow in
    `render`, once a fold has chosen (θ, d).

    :param use_cache: reuse the scan and the verdicts cached in the session
        directory, and store fresh ones.
    :param on_progress: called as (judged_changes, total_changes) while judging.
    :param is_cancelled: polled before each judged change; True stops early.
    :param reuse_verdicts: False asks the model again for every change — what
        the repeatability check needs.
    """
    times_s, sim = scan_similarity(session, use_cache=use_cache)
    return VideoRaw(
        recording_path=Path(session.recording_path),
        session_dir=session.session_dir,
        times_s=times_s,
        sim=sim,
        reuse_verdicts=use_cache and reuse_verdicts,
        on_progress=on_progress,
        is_cancelled=is_cancelled,
    )
