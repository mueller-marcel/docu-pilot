"""
From the result table to the figures: report.analyse turns per-session rows
into the report's bars.
"""

import pytest

from docupilot.config import Settings
from docupilot.evaluation import experiment
from docupilot.evaluation import report as rep
from docupilot.evaluation import statistics as st
from docupilot.evaluation.analysis import subsets

PLAYERS = ("audio", "events", "video")
F1 = {"": 0.1, "audio": 0.6, "events": 0.3, "video": 0.3,
      "audio+events": 0.65, "audio+video": 0.62, "events+video": 0.4,
      "audio+events+video": 0.7}


def result(n_sessions=4):
    """Rows of a run: every session scores F1 per coalition ± a session offset,
    with as many predictions as annotated boundaries (count ratio 1)."""
    settings = Settings(sesoi=0.11, sesoi_source="test")
    res = experiment.ExperimentResult(settings)
    for i in range(n_sessions):
        offset = (i - (n_sessions - 1) / 2) * 0.02
        for coalition in subsets(PLAYERS):
            label = experiment.subset_label(coalition)
            f1 = F1["+".join(sorted(coalition))] + offset
            res.rows.append({
                "session": f"s{i}", "seed": 0, "subset": label, "tau_s": settings.tau_s,
                "threshold": 0.5, "n_predicted": 10, "tp": 5, "fp": 5, "fn": 5,
                "precision": f1, "recall": f1, "f1": f1,
            })
    return res


def test_analyse_produces_the_numbers_of_the_figures():
    r = rep.analyse([], result())
    assert r.n_sessions == 4
    assert r.f1_all == pytest.approx(0.7)
    assert r.count_ratio == pytest.approx(1.0)
    assert r.efficiency_error < 1e-12
    assert r.curve[0] == pytest.approx(0.1) and r.curve[3] == pytest.approx(0.7)
    assert sum(e.est for e in r.steps.values()) == pytest.approx(0.6)
    assert sum(e.est for e in r.shapley.values()) == pytest.approx(0.6)


def test_intervals_are_95_percent_t_intervals_over_sessions():
    r = rep.analyse([], result())
    full = r.subsets[frozenset(PLAYERS)]
    assert (full.est, full.lo, full.hi) == pytest.approx(st.t_interval(list(full.values.values())))


def test_tf1_bars_are_single_modalities_best_first_then_none():
    bars = rep.single_bars(rep.analyse([], result()))
    assert [b[0] for b in bars] == ["Speech", "Events", "Video", rep.BASELINE_NAME]
    assert bars[-1][4] == rep.BASELINE


def test_tf2_bars_are_all_eight_combinations_with_the_empty_one_last():
    bars = rep.subset_bars(rep.analyse([], result()))
    assert len(bars) == 8
    assert bars[0][0] == "E+S+V" and bars[-1][0] == "( )"
    assert [b[1] for b in bars[:-1]] == sorted((b[1] for b in bars[:-1]), reverse=True)


def test_baseline_is_the_mean_f1_without_any_modality():
    r = rep.analyse([], result())
    assert r.baseline == pytest.approx(0.1)
    assert r.baseline == r.subsets[frozenset()].est
    assert r.baseline == pytest.approx(rep.subset_bars(r)[-1][1])
    assert r.baseline == pytest.approx(rep.single_bars(r)[-1][1])
    assert r.sesoi == pytest.approx(0.11)


def test_f1_charts_state_baseline_sesoi_and_sessions_under_the_bars():
    r = rep.analyse([], result())
    line = "v(∅) = 0,100   ·   SESOI = 0,110   ·   4 sessions"
    assert rep.singles_text(r).splitlines()[-1].strip() == line
    assert rep.subsets_text(r).splitlines()[-1].strip() == line
    assert len(rep.subsets_text(r).splitlines()) == 9


def test_f1_chart_hints_name_the_band_and_the_gain_chart_keeps_its_own():
    by_chart = {s.chart: s for s in rep.sections(rep.analyse([], result()))}
    assert "v(∅) ± SESOI" in by_chart["singles"].hint
    assert "v(∅) ± SESOI" in by_chart["subsets"].hint
    assert "±SESOI" in by_chart["steps"].hint
    assert "SESOI" not in by_chart["shapley"].hint


def test_tf3_steps_are_classified_against_the_sesoi():
    r = rep.analyse([], result())
    assert [b[0] for b in rep.step_bars(r)] == ["Modality 1", "Modality 2", "Modality 3"]
    assert r.steps[1].cls(r.sesoi) == st.RELEVANT
    assert r.saturation.k_star in (1, 2, 3)


def test_count_ratio_detects_over_segmentation():
    rows = [{"subset": "audio", "n_predicted": 30, "tp": 8, "fn": 2},
            {"subset": "audio", "n_predicted": 10, "tp": 5, "fn": 5}]
    assert rep.count_ratio(rows, frozenset({"audio"})) == pytest.approx(40 / 20)
