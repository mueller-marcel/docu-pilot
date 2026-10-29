"""
The finished evaluation of one corpus: the report's figures and the numbers
behind them.

    singles     F1 of every single modality and of none — bars, 95 % interval,
                v(∅) ± SESOI band
    subsets     F1 of all eight modality combinations — bars, 95 % interval,
                v(∅) ± SESOI band
    shapley     Shapley value of every modality — bars, 95 % interval
    saturation  saturation curve, mean F1 per number of modalities, with the
                gain written between the points
    steps       gain Δ(k) of the k-th modality — bars, 95 % interval, ±SESOI band

Everything is computed here, without Qt, so the same numbers can be produced by
a script — the figures must be reproducible without a window.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from docupilot.config import Settings
from docupilot.evaluation import analysis, experiment, statistics
from docupilot.evaluation.experiment import ExperimentResult, SessionData
from docupilot.evaluation.statistics import Estimate, SaturationPoint
from docupilot.segmentation import MODALITIES

NAMES = {"audio": "Speech", "events": "Events", "video": "Video"}
LETTERS = {"events": "E", "audio": "S", "video": "V"}
BASELINE_NAME = "no modality (v(∅))"

BAR = "#7c7cf2"
BASELINE = "#9ca3af"

Bar = tuple[str, float, float, float, str]

_EMPTY = frozenset()


@dataclass
class Report:
    """Everything the figures need, computed once off the UI thread."""

    settings: Settings
    players: tuple[str, ...]
    n_sessions: int
    rows: list[dict] = field(default_factory=list)
    subsets: dict[frozenset[str], Estimate] = field(default_factory=dict)
    """F1 of every modality combination, the empty one included."""
    shapley: dict[str, Estimate] = field(default_factory=dict)
    curve: dict[int, float] = field(default_factory=dict)
    steps: dict[int, Estimate] = field(default_factory=dict)
    saturation: SaturationPoint = field(default_factory=lambda: SaturationPoint(0, ()))
    f1_all: float = float("nan")
    count_ratio: float = float("nan")
    """Predicted / annotated boundaries of the full model, pooled over sessions:
    above 1 the model over-segments, below 1 it under-segments."""
    efficiency_error: float = 0.0

    @property
    def tau_s(self) -> float:
        return self.settings.tau_s

    @property
    def sesoi(self) -> float:
        return self.settings.sesoi_value

    @property
    def baseline(self) -> float:
        """v(∅): the mean over the sessions of the F1 without any modality."""
        return self.subsets[_EMPTY].est


def count_ratio(rows: Sequence[dict], subset: frozenset[str]) -> float:
    """Predicted / annotated boundaries of one coalition, pooled over sessions
    and start values."""
    label = experiment.subset_label(subset)
    picked = [r for r in rows if r["subset"] == label]
    annotated = sum(r["tp"] + r["fn"] for r in picked)
    return sum(r["n_predicted"] for r in picked) / annotated if annotated else float("nan")


def analyse(
    data: Sequence[SessionData],
    primary: ExperimentResult,
    players: Sequence[str] = MODALITIES,
) -> Report:
    """
    Turn the result table into the figures.

    :param data: the corpus the run was computed on (the figures need only the
        run's rows; kept so the window's call stays the same).
    :param primary: the run.
    :param players: the modalities.
    """
    settings = primary.settings
    players = tuple(players)
    rows = primary.rows
    paired = experiment.paired_metric(rows, settings.tau_s)
    values = experiment.subset_values(rows, settings.tau_s)
    full = frozenset(players)
    steps = {k: statistics.estimate(f"Δ({k})", v)
             for k, v in statistics.steps_per_session(paired, players).items()}
    return Report(
        settings=settings,
        players=players,
        n_sessions=len(paired.get(full, {})),
        rows=list(rows),
        subsets={s: statistics.estimate(experiment.subset_label(s), v)
                 for s, v in paired.items()},
        shapley={m: statistics.estimate(f"φ {m}", v)
                 for m, v in statistics.shapley_per_session(paired, players).items()},
        curve=analysis.saturation(values, players),
        steps=steps,
        saturation=statistics.saturation_point(steps, settings.sesoi_value),
        f1_all=values.get(full, float("nan")),
        count_ratio=count_ratio(rows, full),
        efficiency_error=max(statistics.efficiency_errors(paired, players).values()),
    )


def single_bars(report: Report) -> list[Bar]:
    """The single modalities, best first, then no modality."""
    bars = []
    for subset, e in sorted(report.subsets.items(), key=lambda kv: -kv[1].est):
        if len(subset) == 1:
            m = next(iter(subset))
            bars.append((NAMES.get(m, m), e.est, e.lo, e.hi, BAR))
    e = report.subsets[_EMPTY]
    return bars + [(BASELINE_NAME, e.est, e.lo, e.hi, BASELINE)]


def combination_label(subset: frozenset[str]) -> str:
    """A combination in the modalities' letters, e.g. "E+S+V"; "( )" for none."""
    return "+".join(sorted(LETTERS.get(m, m) for m in subset)) or "( )"


def subset_bars(report: Report) -> list[Bar]:
    """All eight combinations, best first, the empty one last."""
    ranked = sorted((s for s in report.subsets if s), key=lambda s: -report.subsets[s].est)
    return [
        (combination_label(s), e.est, e.lo, e.hi, BASELINE if not s else BAR)
        for s in (*ranked, _EMPTY) for e in (report.subsets[s],)
    ]


def shapley_bars(report: Report) -> list[Bar]:
    """Shapley value of every modality, strongest first."""
    return [(NAMES.get(m, m), e.est, e.lo, e.hi, BAR)
            for m, e in sorted(report.shapley.items(), key=lambda kv: -kv[1].est)]


def step_bars(report: Report) -> list[Bar]:
    """The gain Δ(k) of the k-th modality, in order of k."""
    return [(f"Modality {k}", e.est, e.lo, e.hi, BAR) for k, e in sorted(report.steps.items())]


def _num(value: float, signed: bool = False) -> str:
    """A number with three decimals and a decimal comma."""
    return f"{value:{'+' if signed else ''}.3f}".replace(".", ",")


def _ci(e: Estimate, signed: bool = False) -> str:
    return f"{_num(e.est, signed)}  [{_num(e.lo, signed)}; {_num(e.hi, signed)}]"


def _saturation_sentence(point: SaturationPoint, n_players: int) -> str:
    if point.k_star >= n_players:
        return f"k* = {point.k_star}: no saturation within the measured range"
    return (f"k* = {point.k_star}: every further gain below SESOI on average "
            "or not statistically significant")


def _band_line(report: Report) -> str:
    """The numbers behind the grey band v(∅) ± SESOI and the sessions it rests on."""
    return (f"   v(∅) = {_num(report.baseline)}   ·   SESOI = {_num(report.sesoi)}   ·   "
            f"{report.n_sessions} sessions")


def singles_text(report: Report) -> str:
    lines = [f"   {label:24s} {_num(v)}  [{_num(lo)}; {_num(hi)}]"
             for label, v, lo, hi, _ in single_bars(report)]
    return "\n".join([*lines, _band_line(report)])


def subsets_text(report: Report) -> str:
    lines = [f"   {label:8s} {_num(v)}  [{_num(lo)}; {_num(hi)}]"
             for label, v, lo, hi, _ in subset_bars(report)]
    return "\n".join([*lines, _band_line(report)])


def shapley_text(report: Report) -> str:
    return "\n".join(f"   {label:12s} {_num(v, True)}  [{_num(lo, True)}; {_num(hi, True)}]"
                     for label, v, lo, hi, _ in shapley_bars(report))


def curve_text(report: Report) -> str:
    return "   " + "   ·   ".join(
        f"{k} modalit{'y' if k == 1 else 'ies'}: {_num(v)}" for k, v in sorted(report.curve.items())
    )


def steps_text(report: Report) -> str:
    """The steps with interval and class against ±SESOI, and k*."""
    lines = [f"   Modality {k}:  {_ci(e, True)}  {e.cls(report.sesoi)}"
             for k, e in sorted(report.steps.items())]
    lines.append("   → " + _saturation_sentence(report.saturation, len(report.players)))
    lines.append(
        f"   SESOI = {_num(report.sesoi)} (one boundary). Classes: R = relevant (CI outside "
        "±SESOI), S = statistically significant, partly below SESOI, SE = significant, but fully "
        "below SESOI, E = negligible, U = undecided."
    )
    return "\n".join(lines)


def status_line(report: Report) -> str:
    """The one-line summary the window shows when a run completes."""
    return (
        f"Done · {report.n_sessions} sessions · τ = {report.tau_s:g} s · "
        f"F1 all modalities {_num(report.f1_all)} · boundaries/annotated "
        f"{report.count_ratio:.2f}".replace(".", ",") + "×"
        + (f"  ·  ⚠ Shapley efficiency error {report.efficiency_error:.1e}"
           if report.efficiency_error > 1e-9 else "")
    )


def header_lines(report: Report, corpus_root: Path | None, created: str) -> list[str]:
    """The metadata block at the top of the PDF report."""
    s = report.settings
    return [
        f"Corpus: {corpus_root if corpus_root else '—'}",
        f"Created: {created}",
        f"Sessions: {report.n_sessions}   ·   Decider: {s.decider}   ·   Leave-one-session-out",
        f"τ = {s.tau_s:g} s   ·   SESOI = {_num(s.sesoi_value)}   ·   Intervals 95 % "
        "(t over the sessions)",
    ]


@dataclass(frozen=True)
class Section:
    """One figure of the report: heading, reading aid, chart and its numbers."""

    title: str
    hint: str
    chart: str | None = None
    text: str | None = None


def sections(report: Report) -> list[Section]:
    """
    The figures in their fixed order — shared by the window and the PDF
    so both show the same headings.
    """
    return [
        Section("Sub-question 1 · Segmentation quality of the single modalities with "
                "confidence interval",
                "Bars = mean F1 score over the sessions, whiskers = 95 % confidence interval; "
                "grey bar = no modality, grey band = v(∅) ± SESOI.",
                chart="singles", text=singles_text(report)),
        Section("Sub-question 2 · F1 scores per modality combination",
                "Bars = mean F1 score, whiskers = 95 % confidence interval; "
                "E = events, S = speech, V = video, ( ) = no modality, "
                "grey band = v(∅) ± SESOI.",
                chart="subsets", text=subsets_text(report)),
        Section("Sub-question 2 · Shapley values with confidence interval",
                "Bars = marginal contribution (gain in F1 score), whiskers = "
                "95 % confidence interval.",
                chart="shapley", text=shapley_text(report)),
        Section("Sub-question 3 · Saturation curve: mean F1 score per number of modalities",
                "Points = mean F1 score of all combinations with k modalities; "
                "the increase between the points.",
                chart="saturation", text=curve_text(report)),
        Section("Sub-question 3 · Gain per added modality with confidence interval",
                "Bars = gain Δ(k), whiskers = 95 % confidence interval, grey band = ±SESOI.",
                chart="steps", text=steps_text(report)),
    ]
