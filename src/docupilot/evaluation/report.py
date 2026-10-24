"""
The finished evaluation of one corpus: the figures of the thesis (chapter 4)
and the numbers behind them.

    TF1   Abb. 4.2  F1 of every single modality and of none — bars, 95 % interval
    TF2   Abb. 4.3  F1 of all eight modality combinations — bars, 95 % interval
          Abb. 4.4  Shapley value of every modality — bars, 95 % interval
    TF3   Abb. 4.5  saturation curve, mean F1 per number of modalities, with the
                    gain written between the points
          Abb. 4.6  gain Δ(k) of the k-th modality — bars, 95 % interval, ±SESOI band

Everything is computed here, without Qt, so the same numbers can be produced by
a script — the figures for the thesis must be reproducible without a window.
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

# The modalities as the thesis names them, and their letters in Abb. 4.3.
NAMES = {"audio": "Sprache", "events": "Ereignisse", "video": "Video"}
LETTERS = {"events": "E", "audio": "S", "video": "V"}
BASELINE_NAME = "keine Modalität (v(∅))"

# Colours of the thesis figures: modalities blue, the uninformed baseline grey.
BAR = "#7c7cf2"
BASELINE = "#9ca3af"

# A bar of a chart: (label, value, ci_low, ci_high, hex colour).
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
    """F1 of every modality combination, the empty one included (TF1, TF2)."""
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


# ── Computation ──────────────────────────────────────────────────────────────

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


# ── Bars for the charts ──────────────────────────────────────────────────────

def single_bars(report: Report) -> list[Bar]:
    """TF1 (Abb. 4.2): the single modalities, best first, then no modality."""
    bars = []
    for subset, e in sorted(report.subsets.items(), key=lambda kv: -kv[1].est):
        if len(subset) == 1:
            m = next(iter(subset))
            bars.append((NAMES.get(m, m), e.est, e.lo, e.hi, BAR))
    e = report.subsets[_EMPTY]
    return bars + [(BASELINE_NAME, e.est, e.lo, e.hi, BASELINE)]


def combination_label(subset: frozenset[str]) -> str:
    """A combination in the letters of Abb. 4.3, e.g. "E+S+V"; "( )" for none."""
    return "+".join(sorted(LETTERS.get(m, m) for m in subset)) or "( )"


def subset_bars(report: Report) -> list[Bar]:
    """TF2 (Abb. 4.3): all eight combinations, best first, the empty one last."""
    ranked = sorted((s for s in report.subsets if s), key=lambda s: -report.subsets[s].est)
    return [
        (combination_label(s), e.est, e.lo, e.hi, BASELINE if not s else BAR)
        for s in (*ranked, _EMPTY) for e in (report.subsets[s],)
    ]


def shapley_bars(report: Report) -> list[Bar]:
    """TF2 (Abb. 4.4): Shapley value of every modality, strongest first."""
    return [(NAMES.get(m, m), e.est, e.lo, e.hi, BAR)
            for m, e in sorted(report.shapley.items(), key=lambda kv: -kv[1].est)]


def step_bars(report: Report) -> list[Bar]:
    """TF3 (Abb. 4.6): the gain Δ(k) of the k-th modality, in order of k."""
    return [(f"{k}. Modalität", e.est, e.lo, e.hi, BAR) for k, e in sorted(report.steps.items())]


# ── Texts ────────────────────────────────────────────────────────────────────

def _num(value: float, signed: bool = False) -> str:
    """A number as the thesis writes it: three decimals, decimal comma."""
    return f"{value:{'+' if signed else ''}.3f}".replace(".", ",")


def _ci(e: Estimate, signed: bool = False) -> str:
    return f"{_num(e.est, signed)}  [{_num(e.lo, signed)}; {_num(e.hi, signed)}]"


def _saturation_sentence(point: SaturationPoint, n_players: int) -> str:
    if point.k_star >= n_players:
        return f"k* = {point.k_star}: keine Sättigung im gemessenen Bereich"
    return (f"k* = {point.k_star}: jeder weitere Zugewinn im Mittel unter SESOI "
            "oder nicht statistisch gesichert")


def singles_text(report: Report) -> str:
    return "\n".join(f"   {label:24s} {_num(v)}  [{_num(lo)}; {_num(hi)}]"
                     for label, v, lo, hi, _ in single_bars(report))


def subsets_text(report: Report) -> str:
    return "\n".join(f"   {label:8s} {_num(v)}  [{_num(lo)}; {_num(hi)}]"
                      for label, v, lo, hi, _ in subset_bars(report))


def shapley_text(report: Report) -> str:
    return "\n".join(f"   {label:12s} {_num(v, True)}  [{_num(lo, True)}; {_num(hi, True)}]"
                     for label, v, lo, hi, _ in shapley_bars(report))


def curve_text(report: Report) -> str:
    return "   " + "   ·   ".join(
        f"{k} Modalität{'' if k == 1 else 'en'}: {_num(v)}" for k, v in sorted(report.curve.items())
    )


def steps_text(report: Report) -> str:
    """TF3: the steps with interval and class against ±SESOI, and k*."""
    lines = [f"   {k}. Modalität:  {_ci(e, True)}  {e.cls(report.sesoi)}"
             for k, e in sorted(report.steps.items())]
    lines.append("   → " + _saturation_sentence(report.saturation, len(report.players)))
    lines.append(
        f"   SESOI = {_num(report.sesoi)} (eine Grenze). Klassen: R = relevant (KI außerhalb "
        "±SESOI), S = statistisch gesichert, teils unter SESOI, SÄ = gesichert, aber ganz unter "
        "SESOI, Ä = vernachlässigbar, U = unklar."
    )
    return "\n".join(lines)


def status_line(report: Report) -> str:
    """The one-line summary the window shows when a run completes."""
    return (
        f"Fertig · {report.n_sessions} Sessions · τ = {report.tau_s:g} s · "
        f"F1 alle Modalitäten {_num(report.f1_all)} · Grenzen/annotiert "
        f"{report.count_ratio:.2f}".replace(".", ",") + "×"
        + (f"  ·  ⚠ Shapley-Effizienzfehler {report.efficiency_error:.1e}"
           if report.efficiency_error > 1e-9 else "")
    )


def header_lines(report: Report, corpus_root: Path | None, created: str) -> list[str]:
    """The metadata block at the top of the PDF report."""
    s = report.settings
    return [
        f"Korpus: {corpus_root if corpus_root else '—'}",
        f"Erstellt: {created}",
        f"Sessions: {report.n_sessions}   ·   Entscheider: {s.decider}   ·   Leave-one-session-out",
        f"τ = {s.tau_s:g} s   ·   SESOI = {_num(s.sesoi_value)}   ·   Intervalle 95 % "
        "(t über die Sessions)",
    ]


@dataclass(frozen=True)
class Section:
    """One figure of the report: heading, reading aid, chart and its numbers."""

    title: str
    hint: str
    chart: str | None = None      # a field of report_view.ReportCharts
    text: str | None = None


def sections(report: Report) -> list[Section]:
    """
    The figures in the order of the thesis — shared by the window and the PDF
    so both show the same headings.
    """
    return [
        Section("Teilfrage 1 · Segmentierungsqualität der Einzelmodalitäten mit "
                "Konfidenzintervall",
                "Balken = mittlerer F1-Wert über die Sessions, Whisker = 95-%-Konfidenzintervall; "
                "grau = keine Modalität.",
                chart="singles", text=singles_text(report)),
        Section("Teilfrage 2 · F1-Werte je Modalitätskombination",
                "Balken = mittlerer F1-Wert, Whisker = 95-%-Konfidenzintervall; "
                "E = Ereignisse, S = Sprache, V = Video, ( ) = keine Modalität.",
                chart="subsets", text=subsets_text(report)),
        Section("Teilfrage 2 · Shapley-Werte mit Konfidenzintervall",
                "Balken = marginaler Beitrag (Zugewinn im F1-Wert), Whisker = "
                "95-%-Konfidenzintervall.",
                chart="shapley", text=shapley_text(report)),
        Section("Teilfrage 3 · Sättigungskurve: mittlerer F1-Wert je Anzahl Modalitäten",
                "Punkte = mittlerer F1-Wert aller Kombinationen mit k Modalitäten; "
                "zwischen den Punkten der Zuwachs.",
                chart="saturation", text=curve_text(report)),
        Section("Teilfrage 3 · Zugewinn je hinzugenommener Modalität mit Konfidenzintervall",
                "Balken = Zugewinn Δ(k), Whisker = 95-%-Konfidenzintervall, graues Band = ±SESOI.",
                chart="steps", text=steps_text(report)),
    ]
