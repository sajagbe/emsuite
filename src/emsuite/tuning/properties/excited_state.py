"""Excited-state property calculators."""

from __future__ import annotations

from emsuite.core.oscillator_strength import as_numpy, oscillator_strength_cpu

from ..constants import HARTREE_TO_EV


def calculate_excited_state_properties(td_obj, triplet: bool, props_to_calc: list[str]) -> dict:
    results: dict = {}
    if td_obj is None or not any(p in props_to_calc for p in ("exe", "osc")):
        return results

    state_prefix = "t" if triplet else "s"

    if "exe" in props_to_calc:
        excitation_energies = as_numpy(td_obj.e) * HARTREE_TO_EV
        for i, energy in enumerate(excitation_energies, 1):
            results[f"{state_prefix}{i}_exe"] = float(energy)

    if "osc" in props_to_calc:
        # CPU extract: GPU td.oscillator_strength() crashes in lib.einsum on CuPy arrays.
        oscillator_strengths = oscillator_strength_cpu(td_obj)
        for i, osc in enumerate(oscillator_strengths, 1):
            results[f"{state_prefix}{i}_osc"] = float(osc)

    return results
