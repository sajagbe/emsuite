"""Extended property registry tests."""

from emsuite.tuning.properties import PROPERTY_CONFIG, setup_calculation


def test_fukui_properties_resolve_dependencies():
    props, calcs = setup_calculation(["fukui_plus", "fukui_minus"])
    assert "ea" in props
    assert "ie" in props
    assert calcs["anion"]
    assert calcs["cation"]


def test_spin_property_registered():
    assert "spin" in PROPERTY_CONFIG
    props, calcs = setup_calculation(["spin"])
    assert "spin" in props
    assert calcs["neutral"]
