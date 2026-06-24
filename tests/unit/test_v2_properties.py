"""v1.2 advanced property registry tests."""

from emsuite.tuning.properties import PROPERTY_CONFIG, setup_calculation


def test_v2_properties_registered():
    for code in (
        "fukui_spa_plus",
        "fukui_spa_minus",
        "freq",
        "stark_homo",
        "stark_gap",
        "eint",
        "h2o",
        "pa",
        "efl_fug",
        "ts_barrier",
    ):
        assert code in PROPERTY_CONFIG


def test_stark_resolves_homo_lumo():
    props, _ = setup_calculation(["stark_gap"])
    assert "stark_homo" in props
    assert "stark_lumo" in props


def test_surface_map_properties_flagged():
    assert PROPERTY_CONFIG["fukui_spa_plus"]["surface_map"] is True
