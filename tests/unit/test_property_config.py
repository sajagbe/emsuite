"""Unit tests for tuning property dependency resolution."""

from emsuite.tuning import PROPERTY_CONFIG, setup_calculation


def test_setup_calculation_single_property():
    props, calcs = setup_calculation(["homo"])
    assert "homo" in props
    assert calcs["neutral"] is True
    assert "cation" not in calcs or calcs.get("cation") is not True


def test_setup_calculation_gap_resolves_dependencies():
    props, calcs = setup_calculation(["gap"])
    assert {"gap", "homo", "lumo"}.issubset(set(props))


def test_setup_calculation_ie_requires_cation():
    props, calcs = setup_calculation(["ie"])
    assert "ie" in props
    assert calcs["cation"] is True


def test_setup_calculation_exe_requires_td():
    props, calcs = setup_calculation(["exe"])
    assert "exe" in props
    assert calcs["td"] is True


def test_setup_calculation_all():
    props, calcs = setup_calculation(["all"])
    assert set(props) == set(PROPERTY_CONFIG.keys())
    assert calcs["neutral"] is True
    assert calcs["cation"] is True
    assert calcs["anion"] is True
    assert calcs["td"] is True
