"""
The matching of predicted to annotated boundaries (metrics.match) and the
annotation's edge rules — the measuring instrument every F1 of the thesis rests on.
"""

import pytest

from docupilot.evaluation.dataset import apply_edge_rules
from docupilot.evaluation.metrics import chance_level, match


def test_prediction_within_tolerance_is_a_hit():
    m = match([13.45], [12.96], 1.0)
    assert (m.tp, m.fp, m.fn) == (1, 0, 0)
    assert m.pairs == [(13.45, 12.96)]
    assert m.f1 == 1.0


def test_distance_equal_to_tolerance_still_counts():
    assert match([10.0], [11.0], 1.0).tp == 1


def test_prediction_beyond_tolerance_is_a_miss_on_both_sides():
    m = match([13.45], [11.90], 1.0)
    assert (m.tp, m.fp, m.fn) == (0, 1, 1)
    assert m.f1 == 0.0


def test_one_to_one_second_prediction_near_the_same_boundary_is_false_positive():
    m = match([13.45], [12.96, 13.80], 1.0)
    assert (m.tp, m.fp, m.fn) == (1, 1, 0)
    # The nearer prediction is kept.
    assert m.pairs == [(13.45, 13.80)]


def test_one_prediction_cannot_satisfy_two_boundaries():
    m = match([10.0, 11.0], [10.5], 1.0)
    assert (m.tp, m.fp, m.fn) == (1, 0, 1)


def test_optimal_assignment_beats_greedy_nearest_neighbour():
    # Greedy would pair 10.9 with 11.0 (distance 0.1) and leave 10.0 without a
    # partner within τ; the optimal assignment finds two hits.
    m = match([10.0, 11.0], [10.9, 11.95], 1.0)
    assert m.tp == 2
    assert sorted(m.pairs) == [(10.0, 10.9), (11.0, 11.95)]


def test_precision_recall_f1_values():
    m = match([5.0, 15.0, 25.0, 35.0], [5.2, 15.4, 50.0], 1.0)
    assert (m.tp, m.fp, m.fn) == (2, 1, 2)
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(0.5)
    assert m.f1 == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))


def test_order_of_inputs_does_not_matter():
    assert match([3.0, 1.0], [1.2, 2.9], 0.5).tp == match([1.0, 3.0], [2.9, 1.2], 0.5).tp == 2


@pytest.mark.parametrize("gt, pred, expected", [
    ([], [1.0], (0, 1, 0)),
    ([1.0], [], (0, 0, 1)),
    ([], [], (0, 0, 0)),
])
def test_empty_sides(gt, pred, expected):
    m = match(gt, pred, 1.0)
    assert (m.tp, m.fp, m.fn) == expected
    assert m.f1 == 0.0


@pytest.mark.parametrize("tau", [0.0, -1.0])
def test_non_positive_tolerance_is_rejected(tau):
    with pytest.raises(ValueError):
        match([1.0], [1.0], tau)


def test_chance_level_is_reproducible_and_zero_without_boundaries():
    gt = [10.0, 25.0, 40.0, 55.0]
    first = chance_level(gt, 60.0, 1.0, n_draws=200, seed=3)
    assert first == chance_level(gt, 60.0, 1.0, n_draws=200, seed=3)
    assert 0.0 < first < 1.0
    assert chance_level([], 60.0, 1.0) == 0.0
    assert chance_level(gt, 0.0, 1.0) == 0.0


def test_edge_rules_drop_boundaries_at_the_edges():
    assert apply_edge_rules([0.1, 0.3, 5.0, 59.8], 60.0) == [0.3, 5.0]


def test_edge_rules_merge_double_marks_into_their_mean():
    assert apply_edge_rules([5.0, 5.05, 9.0], None) == pytest.approx([5.025, 9.0])


def test_edge_rules_sort_and_keep_marks_just_apart():
    assert apply_edge_rules([9.0, 5.0, 5.2], None) == pytest.approx([5.0, 5.2, 9.0])
