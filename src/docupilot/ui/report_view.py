"""
Where a finished Report meets the chart widgets.

The window shows the charts live and the PDF renders them off-screen; both
must display the same figures, so the mapping from report to widget exists
exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from docupilot.evaluation import report as rep
from docupilot.ui.widgets.BarChartWidget import BarChartWidget
from docupilot.ui.widgets.SaturationChartWidget import SaturationChartWidget


@dataclass(frozen=True)
class ReportCharts:
    """The figures of a report, as widgets; field names are the
    `report.Section.chart` names."""

    singles: BarChartWidget
    subsets: BarChartWidget
    shapley: BarChartWidget
    saturation: SaturationChartWidget
    steps: BarChartWidget

    @classmethod
    def create(cls) -> ReportCharts:
        return cls(BarChartWidget(), BarChartWidget(), BarChartWidget(),
                   SaturationChartWidget(), BarChartWidget())

    def by_name(self, name: str):
        """The widget a `report.Section.chart` name refers to."""
        return getattr(self, name)

    def widgets(self) -> list:
        return [getattr(self, f.name) for f in fields(self)]


def apply_report(charts: ReportCharts, report: rep.Report) -> None:
    """Push a report's numbers into the given chart widgets."""
    def band(chart: str) -> dict:
        width, centre = rep.chart_band(report, chart)
        return {"band": width, "band_center": centre}

    charts.singles.set_values(rep.single_bars(report), axis_title="F1 score",
                              **band("singles"))
    charts.subsets.set_values(rep.subset_bars(report), axis_title="F1 score",
                              **band("subsets"))
    charts.shapley.set_values(rep.shapley_bars(report),
                              axis_title="Shapley value (gain in F1 score)",
                              **band("shapley"))
    charts.saturation.set_values(report.curve)
    charts.steps.set_values(rep.step_bars(report), axis_title="Gain Δ(k) in F1 score",
                            **band("steps"))


def build_charts(report: rep.Report) -> ReportCharts:
    """Fresh, parentless chart widgets showing the report — what the PDF renders."""
    charts = ReportCharts.create()
    apply_report(charts, report)
    return charts
