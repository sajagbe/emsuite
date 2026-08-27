"""Property registry and calculation setup."""

from __future__ import annotations

import numpy as np

from ..constants import HARTREE_TO_EV, HARTREE_TO_KCAL
from .excited_state import calculate_excited_state_properties
from .ground_state import calculate_ground_state_properties
from .interaction import interaction_energy_kcal, proton_affinity_kcal
from .stark import compute_stark_properties
from .thermo import calculate_thermo_properties
from .thermo_ext import fugacity_extensions
from .vibrational import fundamental_frequency_cm1

PROPERTY_CONFIG = {
    "gse": {"deps": [], "calc": [], "unit": 1},
    "homo": {"deps": [], "calc": [], "unit": 1},
    "lumo": {"deps": [], "calc": [], "unit": 1},
    "gap": {"deps": ["homo", "lumo"], "calc": [], "unit": 1},
    "dm": {"deps": [], "calc": [], "unit": 1},
    "spin": {"deps": [], "calc": [], "unit": 1},
    "ie": {"deps": [], "calc": ["cation"], "unit": HARTREE_TO_KCAL},
    "ea": {"deps": [], "calc": ["anion"], "unit": HARTREE_TO_KCAL},
    "cp": {"deps": ["ie", "ea"], "calc": [], "unit": HARTREE_TO_KCAL},
    "eng": {"deps": ["cp"], "calc": [], "unit": HARTREE_TO_EV},
    "hard": {"deps": ["ie", "ea"], "calc": [], "unit": HARTREE_TO_EV},
    "efl": {"deps": ["cp", "hard"], "calc": [], "unit": HARTREE_TO_EV},
    "nfl": {"deps": ["efl"], "calc": [], "unit": HARTREE_TO_EV},
    "fukui_plus": {"deps": ["ea"], "calc": ["anion"], "unit": HARTREE_TO_EV},
    "fukui_minus": {"deps": ["ie"], "calc": ["cation"], "unit": HARTREE_TO_EV},
    "freq": {"deps": [], "calc": [], "unit": 1},
    "stark_homo": {"deps": [], "calc": [], "unit": 1},
    "stark_lumo": {"deps": [], "calc": [], "unit": 1},
    "stark_gap": {"deps": ["stark_homo", "stark_lumo"], "calc": [], "unit": 1},
    "eint": {"deps": [], "calc": [], "unit": HARTREE_TO_KCAL},
    "h2o": {"deps": [], "calc": [], "unit": HARTREE_TO_KCAL},
    "pa": {"deps": [], "calc": ["cation"], "unit": HARTREE_TO_KCAL},
    "efl_fug": {"deps": ["efl"], "calc": [], "unit": 1},
    "nfl_fug": {"deps": ["nfl"], "calc": [], "unit": 1},
    "eng_fug": {"deps": ["eng"], "calc": [], "unit": 1},
    "exe": {"deps": [], "calc": ["td"], "unit": 1},
    "osc": {"deps": [], "calc": ["td"], "unit": 1},
}


def setup_calculation(requested_props):
    """Resolve property dependencies and required calculations."""
    if "all" in requested_props:
        requested_props = list(PROPERTY_CONFIG.keys())

    props_needed: set[str] = set()

    def add_deps(prop: str) -> None:
        if prop in props_needed:
            return
        props_needed.add(prop)
        for dep in PROPERTY_CONFIG[prop]["deps"]:
            add_deps(dep)

    for prop in requested_props:
        add_deps(prop)

    calcs_needed: dict[str, bool] = {"neutral": True}
    for prop in props_needed:
        for calc in PROPERTY_CONFIG[prop]["calc"]:
            calcs_needed[calc] = True

    return list(props_needed), calcs_needed


def calculate_all_properties(
    mf,
    anion_mf=None,
    cation_mf=None,
    td_obj=None,
    triplet=False,
    props_to_calc=None,
    probe_coord: np.ndarray | None = None,
    probe_charge: float = 0.0,
):
    """Calculate molecular properties from converged SCF/TD objects."""
    if not props_to_calc:
        return {}

    props = list(props_to_calc)

    results = calculate_ground_state_properties(mf, props)
    results = calculate_thermo_properties(mf, anion_mf, cation_mf, props, partial=results)
    results.update(calculate_excited_state_properties(td_obj, triplet, props))
    results.update(fugacity_extensions(results, props))

    if "fukui_plus" in props and "ea" in results:
        results["fukui_plus"] = results["ea"] * HARTREE_TO_EV
    if "fukui_minus" in props and "ie" in results:
        results["fukui_minus"] = results["ie"] * HARTREE_TO_EV

    if "freq" in props:
        results["freq"] = fundamental_frequency_cm1(mf)

    if "eint" in props and mf is not None:
        results["eint"] = mf.e_tot * HARTREE_TO_KCAL

    if "h2o" in props and mf is not None:
        results["h2o"] = mf.e_tot * HARTREE_TO_KCAL

    if "pa" in props:
        results["pa"] = proton_affinity_kcal(mf, cation_mf)

    stark = compute_stark_properties(mf, probe_coord, probe_charge, props)
    results.update(stark)
    if "stark_gap" in props and "stark_homo" in results and "stark_lumo" in results:
        results["stark_gap"] = results["stark_lumo"] - results["stark_homo"]

    return results


def interaction_effect_kcal(mf_alone, mf_complex) -> float:
    return interaction_energy_kcal(mf_alone, mf_complex)
