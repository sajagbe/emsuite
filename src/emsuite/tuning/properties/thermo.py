"""Thermodynamic and reactivity property calculators."""

from __future__ import annotations


def calculate_thermo_properties(
    mf,
    anion_mf,
    cation_mf,
    props_to_calc: list[str],
    partial: dict | None = None,
) -> dict:
    results = dict(partial or {})

    if "ie" in props_to_calc and cation_mf:
        results["ie"] = cation_mf.e_tot - mf.e_tot
    if "ea" in props_to_calc and anion_mf:
        results["ea"] = mf.e_tot - anion_mf.e_tot

    if "cp" in props_to_calc and all(k in results for k in ("ie", "ea")):
        results["cp"] = -(results["ie"] + results["ea"]) / 2
    if "eng" in props_to_calc and "cp" in results:
        results["eng"] = -results["cp"]
    if "hard" in props_to_calc and all(k in results for k in ("ie", "ea")):
        results["hard"] = (results["ie"] - results["ea"]) / 2
    if "efl" in props_to_calc and all(k in results for k in ("cp", "hard")):
        results["efl"] = results["cp"] ** 2 / (2 * results["hard"]) if results["hard"] != 0 else 0
    if "nfl" in props_to_calc and "efl" in results:
        results["nfl"] = 1 / results["efl"] if results["efl"] != 0 else 0

    return results
