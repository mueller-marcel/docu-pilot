"""
Every setting a segmentation or evaluation run depends on, in one place, each
with where its value comes from.

There is no development session outside the corpus. A parameter is therefore
either fixed before any label was seen (a priori, technical, set), frozen
because re-tuning it would cost model calls, or calibrated inside each outer
fold on that fold's TRAINING sessions only. `PARAMETERS` states which, and the
report prints that table.

`Settings` holds the switches of one run and is what a variant (C12) changes.
It is stored as JSON beside the corpus (`SETTINGS_FILE`) so a run can be
archived together with its configuration, and so the reliability analysis can
write τ and the SESOI back with their provenance.
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
AUDIO_PEAK_MODES: tuple[str, ...] = ("offset", "fraction")
AUDIO_ANCHORS: tuple[str, ...] = ("begin", "end")
EVENT_CHANNEL_SETS: tuple[str, ...] = ("all", "rest")
ASR_ALIGNMENTS: tuple[str, ...] = ("whisper", "whisperx")
CI_METHODS: tuple[str, ...] = ("naive", "nadeau_bengio")
MULTIPLICITY: tuple[str, ...] = ("none", "bonferroni")
TAU_RULES: tuple[str, ...] = ("mean", "p90", "p95")
SESOI_RULES: tuple[str, ...] = ("mean_abs", "sd")

# ── Fixed values (not part of a variant) ──────────────────────────────────────

# Sampling grid of the curves without frames of their own (audio, events), and
# of the untrained rule decider A0. Technical.
GRID_HZ = 50.0

# Candidate grid of the "grid" pool: 0.2 s, far below τ. Technical.
CAND_HZ = 5.0

# Feature windows around a candidate; the same for every curve so the Shapley
# values cannot depend on design effort per modality.
FEATURE_WINDOWS_S: tuple[float, ...] = (0.5, 1.0, 2.0)

# Decision thresholds the inner cross-validation picks from (C4).
THRESHOLD_GRID: tuple[float, ...] = tuple(round(0.05 * k, 2) for k in range(1, 20))

# Folds of the inner GroupKFold that chooses the threshold (C4).
INNER_FOLDS = 5
# Trees of the final forest and of the cheaper inner fits (C4).
FOREST_TREES = 300
FOREST_TREES_INNER = 200

# Tolerances the predictions are scored at; the primary one is `tau_primary_s`.
TAU_SWEEP_S: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)

# Audio: how far a peak may reach BEFORE its sentence. 2τ, a priori (C1c).
PRE_S = 2.0
# Audio: upper bound of the peak, as a share of the sentence's window (C1c).
AUDIO_PEAK_MAX_SHARE = 0.9
# Audio "fraction" mode: where in its window the completion is expected. The
# former `_COMPLETION_POSITION`; only for the sensitivity variant.
AUDIO_PEAK_FRACTION = 0.75

# In-fold calibration (C8).
REST_FULL_CANDIDATES_S: tuple[float, ...] = (4.0, 6.0, 8.0, 10.0, 12.0)
REST_FULL_DEFAULT_S = 8.0
AUDIO_MIN_PAIRS = 10

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
FROZEN = "eingefroren"
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
    Parameter("SPREAD_S, LABEL_TOL_S, NMS_S", "= τ", A_PRIORI,
              "an TAU_PRIMARY gebunden; ändert sich τ, ziehen sie mit"),
    Parameter("PRE_S", "2,0 s", A_PRIORI, "2τ; wie weit ein Audio-Peak vor den Satz reichen darf"),
    Parameter("GRID_HZ, CAND_HZ", "50 / 5 Hz", TECHNICAL, "fest"),
    Parameter("_SETTLE_OFFSET_S, _PIXEL_CHANGE_EPS", "0,2 s / 1,0", SET,
              "fest; nicht getunt, nur als gesetzt dokumentiert"),
    Parameter("ACTIVITY_QUIET, _MIN_DWELL_S", "0,08 / 0,5 s", FROZEN,
              "bestimmen die Dwells und damit die VLM-Aufrufe; laut Code auf einer "
              "Entwicklungssession gesweept (Herkunft: N5) — eingefroren, Sensitivität "
              "nur auf Teilkorpus"),
    Parameter("_MAX_CALLS", "400", TECHNICAL, "fest; Treffer je Session im Bericht"),
    Parameter("AUDIO_ANCHOR, PEAK_OFFSET_S", "–", IN_FOLD,
              "Anker mit kleinerer Streuung, d = Median mit Vorzeichen; < 10 Paare → "
              "Satzbeginn, d = 0"),
    Parameter("REST_FULL_S", "8 s", IN_FOLD,
              "Auswahl aus {4, 6, 8, 10, 12} s über A0 auf {Ereignisse}; Gleichstand → 8 s"),
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

    They only shape how a modality's cached raw result is DRAWN onto its curve,
    never what a model was asked — changing them costs no model call.
    """

    rest_full_s: float = REST_FULL_DEFAULT_S
    audio_anchor: str = "begin"
    peak_offset_s: float = 0.0

    def __post_init__(self) -> None:
        if self.audio_anchor not in AUDIO_ANCHORS:
            raise ValueError(f"audio_anchor = {self.audio_anchor!r}; erlaubt: {AUDIO_ANCHORS}")
        if self.rest_full_s <= 0:
            raise ValueError("rest_full_s muss positiv sein")


# ── Run settings ──────────────────────────────────────────────────────────────

SETTINGS_FILE = "docupilot_settings.json"

_NOT_DERIVED = "gesetzt — noch nicht aus der Wiederholungsannotation (N2) hergeleitet"

# The sensitivity runs of the method comparison (C12). Each entry overrides the
# primary settings; every variant is a full run over all 8 coalitions.
DEFAULT_VARIANTS: tuple[dict[str, Any], ...] = (
    {"decider": "rule"},                  # A0 (C7)
    {"candidate_pool": "union"},          # timing credit (C2)
    {"event_channels": "rest"},           # event modality without C14
)


@dataclass(frozen=True)
class Settings:
    """The switches of one run. `replace(**overrides)` makes a variant."""

    boundary_definition: str = "begin"
    candidate_pool: str = "grid"
    decider: str = "forest"
    burst_pause_s: float = 2.0
    audio_peak_mode: str = "offset"
    event_channels: str = "all"
    activity_quiet: float = 0.08
    asr_alignment: str = "whisper"

    tau_primary_s: float = 1.0
    tau_source: str = _NOT_DERIVED
    sesoi: float = 0.05
    sesoi_source: str = _NOT_DERIVED
    tau_rule: str = "p95"
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
            ("audio_peak_mode", AUDIO_PEAK_MODES),
            ("event_channels", EVENT_CHANNEL_SETS),
            ("asr_alignment", ASR_ALIGNMENTS),
            ("ci_method", CI_METHODS),
            ("multiplicity", MULTIPLICITY),
            ("tau_rule", TAU_RULES),
            ("sesoi_rule", SESOI_RULES),
        ):
            value = getattr(self, name)
            if value not in allowed:
                raise ValueError(f"{name} = {value!r}; erlaubt: {allowed}")
        if self.tau_primary_s <= 0 or self.sesoi <= 0:
            raise ValueError("tau_primary_s und sesoi müssen positiv sein")
        if not 0.0 < self.ci_level < 1.0:
            raise ValueError(f"ci_level = {self.ci_level}; erwartet (0, 1)")
        object.__setattr__(self, "variants", tuple(dict(v) for v in self.variants))

    # Tolerances bound to τ (C8): changing τ moves them along.
    @property
    def spread_s(self) -> float:
        return self.tau_primary_s

    @property
    def label_tol_s(self) -> float:
        return self.tau_primary_s

    @property
    def nms_s(self) -> float:
        return self.tau_primary_s

    def replace(self, **changes: Any) -> Settings:
        return replace(self, **changes)

    def variant(self, overrides: dict[str, Any]) -> Settings:
        """This run with some switches changed; a variant has no variants of its own."""
        return replace(self, **overrides, variants=())

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["variants"] = [dict(v) for v in self.variants]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Settings:
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
