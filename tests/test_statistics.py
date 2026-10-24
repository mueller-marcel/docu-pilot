"""
The statistics behind the figures: the 95 % interval over sessions, the SESOI,
the class rule against ±SESOI and the saturation point k*.
"""

import math

import pytest
from scipy.stats import t

from docupilot.evaluation import statistics as st


def test_t_interval_matches_textbook_formula():
    est, lo, hi = st.t_interval([1.0, 2.0, 3.0, 4.0, 5.0])
    half = t.ppf(0.975, 4) * math.sqrt(2.5) / math.sqrt(5)
    assert est == pytest.approx(3.0)
    assert (lo, hi) == pytest.approx((3.0 - half, 3.0 + half))
    assert half == pytest.approx(1.9632, abs=1e-4)


def test_t_interval_level_is_95_percent_by_default():
    assert st.LEVEL == 0.95
    wide = st.t_interval([1.0, 2.0, 3.0], level=0.99)
    narrow = st.t_interval([1.0, 2.0, 3.0])
    assert wide[1] < narrow[1] and wide[2] > narrow[2]


def test_t_interval_degenerates_below_two_values():
    assert st.t_interval([0.4]) == (0.4, 0.4, 0.4)
    est, lo, hi = st.t_interval([])
    assert math.isnan(est) and math.isnan(lo) and math.isnan(hi)


def test_estimate_orders_sessions_and_carries_the_interval():
    e = st.estimate("x", {"s2": 0.2, "s1": 0.4})
    assert list(e.values) == ["s1", "s2"]
    assert e.est == pytest.approx(0.3)
    assert (e.est, e.lo, e.hi) == pytest.approx(st.t_interval([0.4, 0.2]))


def test_sesoi_is_the_mean_of_one_over_boundaries_per_session():
    value, source = st.sesoi_from_reference({"a": [1.0] * 10, "b": [1.0] * 5})
    assert value == pytest.approx((1 / 10 + 1 / 5) / 2)
    assert "2 sessions" in source


def test_sesoi_is_not_one_over_the_mean_count():
    value, _ = st.sesoi_from_reference({"a": [0.0] * 6, "b": [0.0] * 19})
    assert value == pytest.approx((1 / 6 + 1 / 19) / 2)
    assert value != pytest.approx(1 / 12.5)


def test_sesoi_skips_sessions_without_boundaries():
    value, _ = st.sesoi_from_reference({"a": [0.0] * 4, "b": []})
    assert value == pytest.approx(0.25)


def test_sesoi_needs_a_boundary():
    with pytest.raises(ValueError):
        st.sesoi_from_reference({"a": [], "b": []})


@pytest.mark.parametrize("lo, hi, expected", [
    (0.27, 0.37, st.RELEVANT),
    (-0.40, -0.20, st.RELEVANT),
    (0.02, 0.08, st.SIGNIFICANT_EQUIVALENT),
    (-0.05, 0.05, st.EQUIVALENT),
    (0.09, 0.15, st.SIGNIFICANT),
    (-0.05, 0.20, st.UNDECIDED),
])
def test_classify_five_cases(lo, hi, expected):
    assert st.classify(lo, hi, 0.11) == expected


def test_classify_band_edges_belong_to_the_band():
    assert st.classify(0.0, 0.11, 0.11) == st.EQUIVALENT
    assert st.classify(0.01, 0.11, 0.11) == st.SIGNIFICANT_EQUIVALENT


def test_estimate_class_uses_its_interval():
    e = st.Estimate("Δ(1)", {}, 0.321, 0.273, 0.370)
    assert e.cls(0.110) == st.RELEVANT


def _step(est, lo, hi):
    return st.Estimate("Δ", {}, est, lo, hi)


def test_saturation_after_the_first_modality():
    steps = {1: _step(0.32, 0.27, 0.37), 2: _step(0.05, 0.02, 0.08), 3: _step(0.02, -0.01, 0.05)}
    point = st.saturation_point(steps, 0.11)
    assert point.k_star == 1
    assert point.relevant_steps == (1,)


def test_no_saturation_when_every_step_reaches_the_sesoi():
    steps = {1: _step(0.321, 0.273, 0.370), 2: _step(0.130, 0.107, 0.152),
             3: _step(0.119, 0.091, 0.148)}
    assert st.saturation_point(steps, 0.110).k_star == 3


def test_a_step_needs_both_mean_above_sesoi_and_interval_above_zero():
    assert not st.relevant_gain(_step(0.12, -0.01, 0.25), 0.11)
    assert not st.relevant_gain(_step(0.10, 0.05, 0.15), 0.11)
    assert st.relevant_gain(_step(0.11, 0.01, 0.21), 0.11)


def test_saturation_point_zero_when_nothing_is_relevant():
    steps = {1: _step(0.05, -0.01, 0.11)}
    assert st.saturation_point(steps, 0.11) == st.SaturationPoint(0, ())
