"""Unit tests for tuning effect normalization."""

from emsuite.tuning import normalize_effects


def test_normalize_effects_maps_to_minus_one_one():
    all_effects = [
        {"exe_effect": 1.0},
        {"exe_effect": 3.0},
        {"exe_effect": 5.0},
    ]
    normalized, params = normalize_effects(all_effects, ["exe_effect"])
    assert normalized[0]["exe_effect"] == -1.0
    assert normalized[1]["exe_effect"] == 0.0
    assert normalized[2]["exe_effect"] == 1.0
    assert params["exe_effect"] == (1.0, 5.0)


def test_normalize_effects_constant_values():
    all_effects = [{"gap_effect": 2.0}, {"gap_effect": 2.0}]
    normalized, _ = normalize_effects(all_effects, ["gap_effect"])
    assert normalized[0]["gap_effect"] == 0.0
    assert normalized[1]["gap_effect"] == 0.0
