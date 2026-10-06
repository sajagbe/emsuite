"""physconst: qcelemental + pint backed conversions and VDW radii."""

from emsuite.physconst import (
    ANGSTROM_TO_METER,
    HARTREE_TO_EV,
    HARTREE_TO_KCAL,
    KT_E_TO_V,
    vdw_radius_angstrom,
)


def test_hartree_conversions_are_codata_scale():
    assert abs(HARTREE_TO_EV - 27.2114) < 1e-3
    assert abs(HARTREE_TO_KCAL - 627.509) < 1e-2


def test_angstrom_to_meter():
    assert ANGSTROM_TO_METER == 1e-10


def test_kte_to_volts_near_room_temperature():
    assert abs(KT_E_TO_V - 0.02568) < 5e-5


def test_vdw_radius_mantina_matches_common_bondi():
    assert abs(vdw_radius_angstrom("C") - 1.70) < 1e-6
    assert abs(vdw_radius_angstrom("O") - 1.52) < 1e-6
    # Mantina H is 1.10 Å (Bondi was 1.20)
    assert abs(vdw_radius_angstrom("H") - 1.10) < 1e-6


def test_vdw_radius_missing_uses_fallback():
    assert vdw_radius_angstrom("Sg", missing=1.50) == 1.50
