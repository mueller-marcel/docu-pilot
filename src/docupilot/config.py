"""
Every setting a segmentation or evaluation run depends on, in one place, each
with where its value comes from.

A parameter is either fixed before any label was seen (a priori, technical,
set), frozen because re-tuning it would cost model calls, measured from the
corpus' repeated annotation (τ, SESOI), or calibrated inside each outer fold on
that fold's TRAINING sessions only. `PARAMETERS` states which, and the report
prints that table.

`Settings` holds the switches of one run and is what a variant (C12) changes.
The switches are stored as JSON beside the corpus (`SETTINGS_FILE`) so a run
can be archived together with its configuration. τ and the SESOI are never
stored there: every run measures them from the corpus (experiment.measure_tau,
experiment.measure_sesoi) and carries them with their provenance.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

# ── Allowed values ────────────────────────────────────────────────────────────

BOUNDARY_DEFINITIONS: tuple[str, ...] = ("begin", "end")
CANDIDATE_POOLS: tuple[str, ...] = ("grid", "own", "union")
DECIDERS: tuple[str, ...] = ("rule", "logreg", "forest")
AUDIO_ANCHORS: tuple[str, ...] = ("begin", "end")
EVENT_CHANNEL_SETS: tuple[str, ...] = ("all", "rest")
ASR_ALIGNMENTS: tuple[str, ...] = ("whisper", "whisperx")
CI_METHODS: tuple[str, ...] = ("naive", "nadeau_bengio")
MULTIPLICITY: tuple[str, ...] = ("none", "bonferroni")
SESOI_RULES: tuple[str, ...] = ("mean_abs", "sd")

# ── Fixed values (not part of a variant) ──────────────────────────────────────

# Sampling grid of every evidence curve, E[n] = E(n / GRID_HZ), and of the
# untrained rule decider A0. Technical.
GRID_HZ = 50.0

# Candidates of the "grid" pool per τ: spacing τ/5, so the grid stays far below
# the tolerance whatever τ the retest yields. Technical.
CAND_PER_TAU = 5

# Feature windows around a candidate, in multiples of τ: ±0.5τ, ±τ, ±2τ. The
# same for every curve so the Shapley values cannot depend on design effort per
# modality; the rank feature uses the ±τ window.
FEATURE_WINDOWS_TAU: tuple[float, ...] = (0.5, 1.0, 2.0)

# τ for DISPLAYING a single recording in the feature dialog, which belongs to no
# corpus. Never used by an evaluation: Settings has no τ until it is measured.
DISPLAY_TAU_S = 1.0

# Decision thresholds the inner cross-validation picks from (C4).
THRESHOLD_GRID: tuple[float, ...] = tuple(round(0.05 * k, 2) for k in range(1, 20))

# Folds of the inner GroupKFold that chooses the threshold (C4).
INNER_FOLDS = 5
# Trees of the final forest and of the cheaper inner fits (C4).
FOREST_TREES = 300
FOREST_TREES_INNER = 200

# Tolerances the predictions are scored at besides the measured τ (robustness).
TAU_SWEEP_S: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)

# Audio: how far a peak may lie BEFORE its sentence, P = 2τ; a priori (C1c).
PRE_TAU = 2.0
# Audio: the last window is closed by the session's median gap between
# announcements; with a single sentence there is none and 4τ is used.
LAST_GAP_TAU = 4.0
# Audio: upper bound of the peak, as a share of the sentence's window (C1c).
AUDIO_PEAK_MAX_SHARE = 0.9

# In-fold calibration (C8).
AUDIO_MIN_PAIRS = 10
# Video: noise threshold θ (frame still when its smallest tile SSIM to the
# previous frame is ≥ θ) and minimum rest duration d, chosen per fold by the
# pooled candidate F1 of the change onsets. Ranges: d from three frames up to the
# value of Video2Action, θ from the value of GIFdroid up to nearly 1.
VIDEO_THETA_CANDIDATES: tuple[float, ...] = (0.95, 0.98, 0.99, 0.995, 0.998)
VIDEO_MIN_REST_CANDIDATES_S: tuple[float, ...] = (0.3, 0.5, 0.7, 1.0)
# What the feature dialog draws with — outside an experiment there is no fold.
VIDEO_THETA_DEFAULT = 0.99
VIDEO_MIN_REST_DEFAULT_S = 0.5

# Rule decider A0 (C7): weights of the three event sub-curves in the combined
# event curve — a priori; the class change is the weakest signal (C14).
EVENT_CHANNEL_WEIGHTS: dict[str, float] = {"rest": 1.0, "completion": 1.0, "class_change": 0.5}

# Annotation edge rules of the guideline (N3), applied once in dataset.py.
GT_EDGE_S = 0.3        # boundaries in the first/last 0.3 s are dropped
GT_MERGE_S = 0.1       # boundaries closer than this are merged

# Primary annotation run; run 2 is the blind test-retest repetition (C5).
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

# ── Parameter table (C8) ──────────────────────────────────────────────────────

A_PRIORI = "a priori"
TECHNICAL = "technisch"
SET = "gesetzt"
MEASURED = "gemessen"
IN_FOLD = "in-fold"


@dataclass(frozen=True)
class Parameter:
    """One row of the parameter table in the report's appendix."""

    name: str
    value: str
    kind: str
    handling: str


PARAMETERS: tuple[Parameter, ...] = (
    Parameter("BURST_PAUSE_S", "2,0 s", A_PRIORI,
              "Literaturwert der Schreibprozessforschung (Wengelin 2006); Sensitivität 1/2/3 s"),
    Parameter("τ", "Formel 3.1", MEASURED,
              "mittlere Abweichung gepaarter Grenzen der Wiederholungsannotation, vor dem "
              "Lauf; Paarung bis zur Hälfte des kleinsten Grenzabstands in Lauf 1. Ohne "
              "Wiederholung kein Lauf"),
    Parameter("SESOI", "Regel N5", MEASURED,
              "F1 des vollen Modells gegen Lauf 1 minus gegen Lauf 2, je Retest-Session; "
              "nach dem Primärlauf, vor der Statistik"),
    Parameter("SPREAD_S (σ), LABEL_TOL_S, NMS_S", "= τ", A_PRIORI,
              "an τ gebunden; ändert sich τ, ziehen sie mit. Glocke bei 3σ abgeschnitten"),
    Parameter("Merkmalsfenster, Rang-Fenster, Abstand A0", "±0,5τ/τ/2τ, τ, τ", A_PRIORI,
              "an τ gebunden"),
    Parameter("Kandidatenabstand", "τ/5", A_PRIORI, "an τ gebunden, bleibt weit unter τ"),
    Parameter("PRE_S", "= 2τ", A_PRIORI, "wie weit ein Audio-Gipfel vor dem Satz liegen darf"),
    Parameter("AUDIO_PEAK_MAX_SHARE", "0,9", A_PRIORI, "Audio-Gipfel höchstens 90 % ins Fenster"),
    Parameter("Letztes Audio-Fenster", "Median / 4τ", A_PRIORI,
              "Median der Ansageabstände der Session; bei nur einem Satz 4τ"),
    Parameter("GRID_HZ", "50 Hz", TECHNICAL, "Abtastung der Evidenzkurven"),
    Parameter("_SETTLE_OFFSET_S, _PIXEL_CHANGE_EPS", "0,2 s / 1,0", SET,
              "fest; nicht getunt, nur als gesetzt dokumentiert"),
    Parameter("_MAX_CALLS", "400", TECHNICAL, "fest; Treffer je Session im Bericht"),
    Parameter("AUDIO_ANCHOR, PEAK_OFFSET_S", "–", IN_FOLD,
              "Anker mit kleinerer Streuung, d = Median mit Vorzeichen; < 10 Paare → "
              "Satzbeginn, d = 0"),
    Parameter("Ruhe-Höhe (Ereignisse)", "Rang / n", A_PRIORI,
              "Perzentilrang der Pause vor dem Burst innerhalb der Session; nutzt nur die "
              "belegte Rangfolge (längere Pausen vor Grenzen), parameterfrei"),
    Parameter("θ (Rauschschwelle), d (Mindestdauer)", "–", IN_FOLD,
              "Auswahl aus θ ∈ {0,95; 0,98; 0,99; 0,995; 0,998} × d ∈ {0,3; 0,5; 0,7; 1,0} s "
              "über das gepoolte Kandidaten-F1 der Änderungsbeginne (±τ, 1:1); Gleichstand → "
              "weniger Änderungen, dann längeres d"),
    Parameter("Entscheidungsschwelle", "–", IN_FOLD, "innere GroupKFold (5) über die Trainingssessions"),
    Parameter("DOUBLE_CLICK_S", "0,5 s", A_PRIORI, "Windows-Standard"),
    Parameter("Gewichte A0 (Ruhe : Abschluss : Klassenwechsel)", "1 : 1 : 0,5", A_PRIORI,
              "Klassenwechsel ist das schwächste Signal"),
    Parameter("GT_EDGE_S, GT_MERGE_S", "0,3 s / 0,1 s", A_PRIORI,
              "Randregeln des Annotationsleitfadens (N3)"),
)

@dataclass(frozen=True)
class FoldParams:
    """
    The parameters calibrated inside each outer fold (C8), at their defaults.

    Audio: they only shape how a cached raw result is DRAWN onto its curve. Video: θ and d decide which screen changes exist and therefore which
    BEFORE|AFTER pairs the VLM judges; verdicts are cached per pair, so a pair
    chosen again costs no model call.
    """

    audio_anchor: str = "begin"
    peak_offset_s: float = 0.0
    video_theta: float = VIDEO_THETA_DEFAULT
    video_min_rest_s: float = VIDEO_MIN_REST_DEFAULT_S

    def __post_init__(self) -> None:
        if self.audio_anchor not in AUDIO_ANCHORS:
            raise ValueError(f"audio_anchor = {self.audio_anchor!r}; erlaubt: {AUDIO_ANCHORS}")
        if not 0.0 < self.video_theta < 1.0:
            raise ValueError(f"video_theta = {self.video_theta}; erwartet (0, 1)")
        if self.video_min_rest_s <= 0:
            raise ValueError("video_min_rest_s muss positiv sein")


# ── Run settings ──────────────────────────────────────────────────────────────

SETTINGS_FILE = "docupilot_settings.json"

# Measured per run from the corpus, never read from or written to the file.
_MEASURED_FIELDS: tuple[str, ...] = ("tau_primary_s", "tau_source", "sesoi", "sesoi_source")

# The sensitivity runs of the method comparison (C12). Each entry overrides the
# primary settings; every variant is a full run over all 8 coalitions.
DEFAULT_VARIANTS: tuple[dict[str, Any], ...] = (
    {"decider": "rule"},                  # A0 (C7)
    {"candidate_pool": "union"},          # timing credit (C2)
    {"event_channels": "rest"},           # event modality without C14
)


@dataclass(frozen=True)
class Settings:
    """
    The switches of one run. `replace(**overrides)` makes a variant.

    τ and the SESOI are None until a run has measured them from the corpus;
    everything bound to τ refuses to answer before that.
    """

    boundary_definition: str = "begin"
    candidate_pool: str = "grid"
    decider: str = "forest"
    burst_pause_s: float = 2.0
    event_channels: str = "all"
    asr_alignment: str = "whisper"

    tau_primary_s: float | None = None
    tau_source: str = ""
    sesoi: float | None = None
    sesoi_source: str = ""
    sesoi_rule: str = "mean_abs"

    ci_level: float = 0.95
    ci_method: str = "nadeau_bengio"
    multiplicity: str = "none"
    bootstrap_check: bool = False
    seed: int = 0

    variants: tuple[dict[str, Any], ...] = field(default=DEFAULT_VARIANTS)

    def __post_init__(self) -> None:
        for name, allowed in (
            ("boundary_definition", BOUNDARY_DEFINITIONS),
            ("candidate_pool", CANDIDATE_POOLS),
            ("decider", DECIDERS),
            ("event_channels", EVENT_CHANNEL_SETS),
            ("asr_alignment", ASR_ALIGNMENTS),
            ("ci_method", CI_METHODS),
            ("multiplicity", MULTIPLICITY),
            ("sesoi_rule", SESOI_RULES),
        ):
            value = getattr(self, name)
            if value not in allowed:
                raise ValueError(f"{name} = {value!r}; erlaubt: {allowed}")
        for name in ("tau_primary_s", "sesoi"):
            value = getattr(self, name)
            if value is not None and not value > 0:
                raise ValueError(f"{name} = {value}; muss positiv sein")
        if not 0.0 < self.ci_level < 1.0:
            raise ValueError(f"ci_level = {self.ci_level}; erwartet (0, 1)")
        object.__setattr__(self, "variants", tuple(dict(v) for v in self.variants))

    @property
    def tau_s(self) -> float:
        """The measured τ; raises before it was measured."""
        if self.tau_primary_s is None:
            raise ValueError("τ ist noch nicht aus der Wiederholungsannotation gemessen")
        return self.tau_primary_s

    @property
    def sesoi_value(self) -> float:
        """The measured SESOI; raises before it was measured."""
        if self.sesoi is None:
            raise ValueError("SESOI ist noch nicht aus der Wiederholungsannotation gemessen")
        return self.sesoi

    # Sizes bound to τ (C8): changing τ moves them along.
    @property
    def spread_s(self) -> float:
        """σ of the bell around a point hint."""
        return self.tau_s

    @property
    def label_tol_s(self) -> float:
        return self.tau_s

    @property
    def nms_s(self) -> float:
        return self.tau_s

    @property
    def pre_s(self) -> float:
        return PRE_TAU * self.tau_s

    @property
    def last_gap_s(self) -> float:
        return LAST_GAP_TAU * self.tau_s

    @property
    def cand_spacing_s(self) -> float:
        return self.tau_s / CAND_PER_TAU

    @property
    def feature_windows_s(self) -> tuple[float, ...]:
        return tuple(k * self.tau_s for k in FEATURE_WINDOWS_TAU)

    def replace(self, **changes: Any) -> Settings:
        return replace(self, **changes)

    def variant(self, overrides: dict[str, Any]) -> Settings:
        """This run with some switches changed; a variant has no variants of its own."""
        return replace(self, **overrides, variants=())

    def to_dict(self) -> dict[str, Any]:
        """The switches, without the measured τ and SESOI."""
        data = asdict(self)
        for name in _MEASURED_FIELDS:
            del data[name]
        data["variants"] = [dict(v) for v in self.variants]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Settings:
        measured = sorted(set(data) & set(_MEASURED_FIELDS))
        if measured:
            raise ValueError(
                f"{', '.join(measured)} lassen sich nicht setzen — τ und SESOI werden bei "
                "jedem Lauf aus der Wiederholungsannotation gemessen."
            )
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unbekannte Einstellungen: {', '.join(unknown)}")
        values = dict(data)
        if "variants" in values:
            values["variants"] = tuple(values["variants"])
        return cls(**values)


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
