"""Thermodynamic and reactivity property calculators."""

from __future__ import annotations

from ..constants import HARTREE_TO_EV, HARTREE_TO_KCAL


def calculate_thermo_properties(
    mf,
    anion_mf,
    cation_mf,
    props_to_calc: list[str],
    partial: dict | None = None,
) -> dict:
    """Conceptual-DFT properties from neutral/cation/anion SCF energies.

    Everything is derived in Hartree first, then converted to the documented
    output units (README property table): ie, ea, cp in kcal/mol; eng, hard,
    efl in eV; nfl = 1/efl in 1/eV.
    """
    results = dict(partial or {})
    ha: dict[str, float] = {}

    if "ie" in props_to_calc and cation_mf:
        ha["ie"] = cation_mf.e_tot - mf.e_tot
    if "ea" in props_to_calc and anion_mf:
        ha["ea"] = mf.e_tot - anion_mf.e_tot

    if "cp" in props_to_calc and all(k in ha for k in ("ie", "ea")):
        ha["cp"] = -(ha["ie"] + ha["ea"]) / 2
    if "eng" in props_to_calc and "cp" in ha:
        ha["eng"] = -ha["cp"]
    if "hard" in props_to_calc and all(k in ha for k in ("ie", "ea")):
        ha["hard"] = (ha["ie"] - ha["ea"]) / 2
    if "efl" in props_to_calc and all(k in ha for k in ("cp", "hard")):
        ha["efl"] = ha["cp"] ** 2 / (2 * ha["hard"]) if ha["hard"] != 0 else 0

    for key in ("ie", "ea", "cp"):
        if key in ha:
            results[key] = ha[key] * HARTREE_TO_KCAL
    for key in ("eng", "hard", "efl"):
        if key in ha:
            results[key] = ha[key] * HARTREE_TO_EV

    if "nfl" in props_to_calc and "efl" in results:
        results["nfl"] = 1 / results["efl"] if results["efl"] != 0 else 0

    return results
