"""
Shapley values, the saturation curve and the gains Δ(k): on whole
games and per session, as the report computes them.
"""

import pytest

from docupilot.evaluation import analysis
from docupilot.evaluation import statistics as st

PLAYERS = ("audio", "events", "video")
S, E, V = "audio", "events", "video"


def game(**by_label):
    """Coalition values from keyword labels: e.g. none=0.1, S=0.6, ES=0.66."""
    names = {"S": S, "E": E, "V": V}
    return {
        frozenset() if label == "none" else frozenset(names[c] for c in label): value
        for label, value in by_label.items()
    }


EXAMPLE = game(none=0.117, S=0.636, E=0.339, V=0.339, ES=0.662, SV=0.646, EV=0.395, ESV=0.687)


def test_additive_game_gives_each_modality_its_own_contribution():
    base, add = 0.1, {S: 0.5, E: 0.2, V: 0.1}
    values = {c: base + sum(add[p] for p in c) for c in analysis.subsets(PLAYERS)}
    assert analysis.shapley(values, PLAYERS) == pytest.approx(add)


def test_efficiency_values_sum_to_full_minus_empty():
    phi = analysis.shapley(EXAMPLE, PLAYERS)
    assert sum(phi.values()) == pytest.approx(0.687 - 0.117)
    assert analysis.efficiency_error(EXAMPLE, PLAYERS, phi) == pytest.approx(0.0, abs=1e-12)


def test_shapley_values_of_the_example_game():
    phi = analysis.shapley(EXAMPLE, PLAYERS)
    assert phi[S] == pytest.approx(0.375, abs=5e-4)
    assert phi[E] == pytest.approx(0.101, abs=5e-4)
    assert phi[V] == pytest.approx(0.093, abs=5e-4)


def test_symmetric_players_get_equal_values_and_a_dummy_gets_zero():
    values = game(none=0.0, S=0.0, E=0.3, V=0.3, ES=0.3, SV=0.3, EV=0.5, ESV=0.5)
    phi = analysis.shapley(values, PLAYERS)
    assert phi[S] == pytest.approx(0.0)
    assert phi[E] == pytest.approx(phi[V])


def test_redundant_modalities_share_their_contribution():
    # E and V see the same thing: each alone +0.2, together still +0.2.
    values = game(none=0.1, S=0.1, E=0.3, V=0.3, ES=0.3, SV=0.3, EV=0.3, ESV=0.3)
    phi = analysis.shapley(values, PLAYERS)
    assert phi[E] == pytest.approx(0.1) and phi[V] == pytest.approx(0.1)


def test_missing_coalition_is_an_error():
    values = dict(EXAMPLE)
    del values[frozenset({E, V})]
    with pytest.raises(KeyError):
        analysis.shapley(values, PLAYERS)


def test_full_factorial_design_has_eight_coalitions():
    coalitions = analysis.subsets(PLAYERS)
    assert len(coalitions) == 8 and len(set(coalitions)) == 8
    assert coalitions[0] == frozenset() and coalitions[-1] == frozenset(PLAYERS)


def test_saturation_curve_is_the_mean_per_size():
    curve = analysis.saturation(EXAMPLE, PLAYERS)
    assert curve[0] == pytest.approx(0.117)
    assert curve[1] == pytest.approx((0.636 + 0.339 + 0.339) / 3)
    assert curve[2] == pytest.approx((0.662 + 0.646 + 0.395) / 3)
    assert curve[3] == pytest.approx(0.687)


def test_gains_of_the_example_game():
    gains = analysis.marginal_gain(analysis.saturation(EXAMPLE, PLAYERS))
    assert gains == pytest.approx({1: 0.321, 2: 0.130, 3: 0.119}, abs=5e-4)


def test_gains_add_up_to_full_minus_empty():
    gains = analysis.marginal_gain(analysis.saturation(EXAMPLE, PLAYERS))
    assert sum(gains.values()) == pytest.approx(0.687 - 0.117)


def per_session():
    """Two sessions; the second is the first shifted by +0.1 in every coalition."""
    s1 = EXAMPLE
    s2 = {c: v + 0.1 for c, v in EXAMPLE.items()}
    return {c: {"s1": s1[c], "s2": s2[c]} for c in EXAMPLE}


def test_shapley_per_session_is_computed_per_session():
    phi = st.shapley_per_session(per_session(), PLAYERS)
    whole = analysis.shapley(EXAMPLE, PLAYERS)
    # A constant shift moves v(∅) too, so no modality gains from it.
    for p in PLAYERS:
        assert phi[p]["s1"] == pytest.approx(whole[p])
        assert phi[p]["s2"] == pytest.approx(whole[p])


def test_steps_per_session_match_the_curve_of_each_session():
    steps = st.steps_per_session(per_session(), PLAYERS)
    gains = analysis.marginal_gain(analysis.saturation(EXAMPLE, PLAYERS))
    for k in (1, 2, 3):
        assert steps[k]["s1"] == pytest.approx(gains[k])
        assert steps[k]["s2"] == pytest.approx(gains[k])


def test_efficiency_errors_are_zero_per_session():
    assert max(st.efficiency_errors(per_session(), PLAYERS).values()) < 1e-12


def test_sessions_must_be_scored_on_the_same_set():
    table = per_session()
    table[frozenset()] = {"s1": 0.1}
    with pytest.raises(KeyError):
        st.steps_per_session(table, PLAYERS)
