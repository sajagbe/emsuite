"""Core property registry tests (advanced modules removed)."""

from emsuite.tuning.properties import PROPERTY_CONFIG

CORE = {
    "gse",
    "homo",
    "lumo",
    "gap",
    "dm",
    "spin",
    "ie",
    "ea",
    "cp",
    "eng",
    "hard",
    "efl",
    "nfl",
    "fukui_plus",
    "fukui_minus",
    "exe",
    "osc",
}


def test_core_properties_registered():
    assert set(PROPERTY_CONFIG) == CORE


def test_advanced_properties_removed():
    for code in (
        "freq",
        "stark_homo",
        "stark_gap",
        "eint",
        "h2o",
        "pa",
        "efl_fug",
    ):
        assert code not in PROPERTY_CONFIG
