"""
Where the report meets the chart widgets: the F1 charts get the v(∅) ± SESOI
band, the gain chart keeps its band around zero.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from docupilot.evaluation import report as rep  # noqa: E402
from docupilot.ui import report_view  # noqa: E402

from test_report import result  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_f1_charts_get_the_band_around_the_baseline(app):
    report = rep.analyse([], result())
    charts = report_view.build_charts(report)
    for chart in (charts.singles, charts.subsets):
        assert chart._band == pytest.approx(report.sesoi)
        assert chart._band_center == pytest.approx(report.baseline)
    assert charts.steps._band == pytest.approx(report.sesoi)
    assert charts.steps._band_center == 0.0
    assert charts.shapley._band is None


def test_charts_render_with_the_band(app):
    charts = report_view.build_charts(rep.analyse([], result()))
    for chart in charts.widgets():
        chart.resize(700, 400)
        assert not chart.grab().isNull()
