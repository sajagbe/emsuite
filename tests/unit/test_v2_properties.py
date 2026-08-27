"""v1.2 advanced property registry tests."""

from emsuite.tuning.properties import PROPERTY_CONFIG, setup_calculation


def test_v2_properties_registered():
    for code in (
        "freq",
        "stark_homo",
        "stark_gap",
        "eint",
        "h2o",
        "pa",
        "efl_fug",
    ):
        assert code in PROPERTY_CONFIG


def test_stark_resolves_homo_lumo():
    props, _ = setup_calculation(["stark_gap"])
    assert "stark_homo" in props
    assert "stark_lumo" in props
