"""
Every setting a segmentation or evaluation run depends on, in one place, each
with where its value comes from.

A parameter is either fixed before any label was seen (a priori, technical,
set), frozen because re-tuning it would cost model calls, measured from the
corpus' annotation (the SESOI from the reference), or
calibrated inside each outer fold on
that fold's TRAINING sessions only. The comment above each value says which.

`Settings` holds the switches of one run.
The switches are stored as JSON beside the corpus (`SETTINGS_FILE`) so a run
can be archived together with its configuration. The SESOI is never stored
there: every run measures it from the corpus (experiment.measure_sesoi) and
carries it with its provenance.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

# ── Allowed values ────────────────────────────────────────────────────────────

BOUNDARY_DEFINITIONS: tuple[str, ...] = ("begin", "end")
CANDIDATE_POOLS: tuple[str, ...] = ("grid", "own", "union")
DECIDERS: tuple[str, ...] = ("rule", "logreg", "forest")
EVENT_CHANNEL_SETS: tuple[str, ...] = ("all", "rest")
ASR_ALIGNMENTS: tuple[str, ...] = ("whisper", "whisperx")

# ── Fixed values ──────────────────────────────────────

# Sampling grid of every evidence curve, E[n] = E(n / GRID_HZ), and of the
# untrained rule decider A0. Technical.
GRID_HZ = 50.0

# Tolerance τ of the evaluation: a prediction within ±τ of an annotated boundary
# is a hit. A priori, from the GEBD protocol (consistency F1 over 0.2 … 1.0 s,
# Shou et al. 2021) — subtask boundaries in screencasts cannot be placed more
# finely (ToolScape: hits 2.7 s off on average, Kim et al. 2014). Can be changed
# in the settings file; the robustness sweep scores the other tolerances anyway.
TOLERANCE_S = 1.0

# Radius of the non-maximum suppression after the decider: of two predicted
# boundaries closer than this only the stronger one survives. A priori from the
# annotation's own granularity — workflow steps follow each other no closer than
# 3 s, so one sub-step never produces two boundaries.
NMS_S = 2.0

# Candidates of the "grid" pool per τ: spacing τ/5, so the grid stays far below
# the tolerance whatever τ is set to. Technical.
CAND_PER_TAU = 5

# Feature windows around a candidate, in multiples of τ: ±0.5τ, ±τ, ±2τ. The
# same for every curve so the Shapley values cannot depend on design effort per
# modality; the rank feature uses the ±τ window.
FEATURE_WINDOWS_TAU: tuple[float, ...] = (0.5, 1.0, 2.0)

# τ for DISPLAYING a single recording in the feature dialog.
DISPLAY_TAU_S = TOLERANCE_S

# Decision thresholds the inner cross-validation picks from (C4).
THRESHOLD_GRID: tuple[float, ...] = tuple(round(0.05 * k, 2) for k in range(1, 20))

# Folds of the inner GroupKFold that chooses the threshold (C4).
INNER_FOLDS = 5

# The forest's size (MA 3.3.1): grown in steps of FOREST_TREES_STEP until the
# error on held-out training sessions changes by less than TREES_STABLE_TOL for
# TREES_STABLE_STEPS steps in a row, at most FOREST_TREES_MAX. Found once per
# outer fold on the full coalition and then used for every coalition, seed and
# inner fit, so all coalitions run with the same settings.
#
# Held-out SESSIONS, not out-of-bag rows: neighbouring grid candidates lie τ/5
# apart and are near copies, so a row's out-of-bag vote comes from trees that
# saw its neighbours — measured on this corpus the OOB probabilities rank the
# boundaries worse than chance (AUC < 0.4) while held-out sessions give 0.8.
FOREST_TREES_STEP = 50
FOREST_TREES_MAX = 300
TREES_HOLDOUT_SHARE = 0.2
TREES_STABLE_TOL = 0.001          # Brier score
TREES_STABLE_STEPS = 2
# TECHNICAL. Leaf size and bootstrap share of every tree: regularise the
# forest on ~10⁵ strongly autocorrelated grid rows and keep a fit to seconds.
FOREST_MIN_SAMPLES_LEAF = 5
FOREST_MAX_SAMPLES = 0.3

# In-fold calibration (C8).
AUDIO_MIN_PAIRS = 10
# Video: noise threshold θ (frame still when its smallest tile SSIM to the
# previous frame is ≥ θ) and minimum rest duration d, chosen per fold by the
# pooled candidate F1 of the change onsets. The ranges enclose the optimum on
# both sides — the former grid (θ ≥ 0.95, d ≤ 1 s) put every fold in its corner.
VIDEO_THETA_CANDIDATES: tuple[float, ...] = (0.85, 0.9, 0.95, 0.98)
VIDEO_MIN_REST_CANDIDATES_S: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0)
# What the feature dialog draws with — outside an experiment there is no fold.
VIDEO_THETA_DEFAULT = 0.9
VIDEO_MIN_REST_DEFAULT_S = 2.0

# Rule decider A0 (C7): weights of the three event sub-curves in the combined
# event curve — a priori; the class change is the weakest signal (C14).
EVENT_CHANNEL_WEIGHTS: dict[str, float] = {"rest": 1.0, "completion": 1.0, "class_change": 0.5}

# Annotation edge rules of the guideline (N3), applied once in dataset.py.
GT_EDGE_S = 0.3        # boundaries in the first/last 0.3 s are dropped
GT_MERGE_S = 0.1       # boundaries closer than this are merged

# Primary annotation run, the reference; later runs are blind repetitions.
PRIMARY_RUN = 1

# Version of the annotation guideline (N3), stored with every boundary. Bump it
# whenever the guideline changes; the annotation window offers it as default.
GUIDELINE_VERSION = "1"

# ── Key classes of the event modality (C14) ───────────────────────────────────
#
# Keys as InputRecorder writes them (`str(key.char)` or `str(Key.x)`), with a
# held Ctrl/Alt/Cmd prefixed as "ctrl+" / "alt+" / "cmd+". Keys listed here are
# classified by the table; any other printable single character is "character",
# any other combination with a modifier is "shortcut", and whatever is left is
# "shortcut" as well but counted as UNKNOWN in the report.

KEY_CLASSES: dict[str, tuple[str, ...]] = {
    "completion": ("Key.enter", "ctrl+Key.enter", "ctrl+s"),
    "abort": ("Key.esc", "ctrl+z"),
    "navigation": (
        "Key.tab", "Key.up", "Key.down", "Key.left", "Key.right",
        "Key.home", "Key.end", "Key.page_up", "Key.page_down",
    ),
    "character": ("Key.space", "Key.backspace", "Key.delete"),
    "shortcut": (
        *(f"Key.f{i}" for i in range(1, 13)),
        "Key.insert", "Key.print_screen", "Key.menu", "Key.pause", "Key.num_lock",
        "Key.scroll_lock", "Key.media_play_pause", "Key.media_volume_up",
        "Key.media_volume_down", "Key.media_volume_mute",
    ),
}

# Pressed alone these only change what the next key means; they are no action
# of their own and do not enter the class histogram.
MODIFIER_KEYS: dict[str, tuple[str, ...]] = {
    "ctrl": ("Key.ctrl", "Key.ctrl_l", "Key.ctrl_r"),
    "alt": ("Key.alt", "Key.alt_l", "Key.alt_r"),
    "cmd": ("Key.cmd", "Key.cmd_l", "Key.cmd_r"),
    "alt_gr": ("Key.alt_gr",),
    "shift": ("Key.shift", "Key.shift_l", "Key.shift_r", "Key.caps_lock"),
}

# Two left clicks closer than this are one double click. The Windows default,
# a priori.
DOUBLE_CLICK_S = 0.5


@dataclass(frozen=True)
class FoldParams:
    """
    The parameters calibrated inside each outer fold (C8), at their defaults.

    Audio: they only shape how a cached raw result is DRAWN onto its curve. Video: θ and d decide which screen changes exist and therefore which
    BEFORE|AFTER pairs the VLM judges; verdicts are cached per pair, so a pair
    chosen again costs no model call.
    """

    audio_peak_offset_s: float = 0.0
    """How far after its start a sentence's bell peaks: the announcement's lead."""
    video_theta: float = VIDEO_THETA_DEFAULT
    video_min_rest_s: float = VIDEO_MIN_REST_DEFAULT_S

    def __post_init__(self) -> None:
        if self.audio_peak_offset_s < 0.0:
            raise ValueError(f"audio_peak_offset_s = {self.audio_peak_offset_s}; erwartet ≥ 0")
        if not 0.0 < self.video_theta < 1.0:
            raise ValueError(f"video_theta = {self.video_theta}; erwartet (0, 1)")
        if self.video_min_rest_s <= 0:
            raise ValueError("video_min_rest_s muss positiv sein")


# ── Run settings ──────────────────────────────────────────────────────────────

SETTINGS_FILE = "docupilot_settings.json"

# Measured per run from the corpus, never read from or written to the file.
_MEASURED_FIELDS: tuple[str, ...] = ("sesoi", "sesoi_source")



@dataclass(frozen=True)
class Settings:
    """
    The switches of one run.

    The SESOI is None until a run has measured it from the corpus.
    """

    boundary_definition: str = "begin"
    candidate_pool: str = "grid"
    decider: str = "forest"
    burst_pause_s: float = 2.0
    event_channels: str = "rest"
    asr_alignment: str = "whisper"

    tau_primary_s: float = TOLERANCE_S
    tau_source: str = "a priori, GEBD-Protokoll"
    sesoi: float | None = None
    sesoi_source: str = ""

    seed: int = 0
    # Start values of the primary run; F1 per session is averaged over them
    # before any statistic, which removes the forest's own randomness from the
    # session-to-session spread.
    n_seeds: int = 3

    def __post_init__(self) -> None:
        for name, allowed in (
            ("boundary_definition", BOUNDARY_DEFINITIONS),
            ("candidate_pool", CANDIDATE_POOLS),
            ("decider", DECIDERS),
            ("event_channels", EVENT_CHANNEL_SETS),
            ("asr_alignment", ASR_ALIGNMENTS),
        ):
            value = getattr(self, name)
            if value not in allowed:
                raise ValueError(f"{name} = {value!r}; erlaubt: {allowed}")
        for name in ("tau_primary_s", "sesoi"):
            value = getattr(self, name)
            if value is not None and not value > 0:
                raise ValueError(f"{name} = {value}; muss positiv sein")
        if self.n_seeds < 1:
            raise ValueError(f"n_seeds = {self.n_seeds}; mindestens 1")

    @property
    def tau_s(self) -> float:
        """The tolerance τ."""
        return self.tau_primary_s

    @property
    def sesoi_value(self) -> float:
        """The measured SESOI; raises before it was measured."""
        if self.sesoi is None:
            raise ValueError("SESOI ist noch nicht aus der Referenzannotation bestimmt")
        return self.sesoi

    # Sizes bound to τ (C8): changing τ moves them along.
    @property
    def spread_s(self) -> float:
        """σ of the bell around a point hint: τ/3, so the bell, cut off at 3σ,
        is exactly as wide as the tolerance window ±τ (MA 3.2.2)."""
        return self.tau_s / 3.0

    @property
    def label_tol_s(self) -> float:
        return self.tau_s

    @property
    def nms_s(self) -> float:
        return NMS_S

    @property
    def cand_spacing_s(self) -> float:
        return self.tau_s / CAND_PER_TAU

    @property
    def feature_windows_s(self) -> tuple[float, ...]:
        return tuple(k * self.tau_s for k in FEATURE_WINDOWS_TAU)

    def replace(self, **changes: Any) -> Settings:
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """The switches, without the measured SESOI."""
        data = asdict(self)
        for name in _MEASURED_FIELDS:
            del data[name]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Settings:
        measured = sorted(set(data) & set(_MEASURED_FIELDS))
        if measured:
            raise ValueError(
                f"{', '.join(measured)} lässt sich nicht setzen — die SESOI wird bei "
                "jedem Lauf aus der Annotation gemessen."
            )
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unbekannte Einstellungen: {', '.join(unknown)}")
        return cls(**data)


def load_settings(path: Path) -> Settings:
    """Settings from a JSON file; the defaults when the file does not exist."""
    if not path.exists():
        return Settings()
    return Settings.from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_settings(settings: Settings, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(settings.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
