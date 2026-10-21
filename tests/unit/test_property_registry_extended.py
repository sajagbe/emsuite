"""Extended property registry tests."""

from emsuite.tuning.properties import PROPERTY_CONFIG, setup_calculation


def test_spin_property_registered():
    assert "spin" in PROPERTY_CONFIG
    props, calcs = setup_calculation(["spin"])
    assert "spin" in props
    assert calcs["neutral"]
