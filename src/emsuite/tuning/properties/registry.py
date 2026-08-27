"""Property registry and calculation setup."""

from __future__ import annotations

from ..constants import HARTREE_TO_EV, HARTREE_TO_KCAL
from .excited_state import calculate_excited_state_properties
from .ground_state import calculate_ground_state_properties
from .thermo import calculate_thermo_properties

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
    "exe": {"deps": [], "calc": ["td"], "unit": 1},
    "osc": {"deps": [], "calc": ["td"], "unit": 1},
}


def setup_calculation(requested_props):
    """Resolve property dependencies and required calculations."""
    if "all" in requested_props:
        requested_props = list(PROPERTY_CONFIG.keys())
    else:
        unknown = [prop for prop in requested_props if prop not in PROPERTY_CONFIG]
        if unknown:
            raise KeyError(
                f"Unknown property {unknown[0]!r}. "
                f"Known properties: {', '.join(sorted(PROPERTY_CONFIG))}"
            )

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
):
    """Calculate molecular properties from converged SCF/TD objects."""
    if not props_to_calc:
        return {}

    props = list(props_to_calc)

    results = calculate_ground_state_properties(mf, props)
    results = calculate_thermo_properties(mf, anion_mf, cation_mf, props, partial=results)
    results.update(calculate_excited_state_properties(td_obj, triplet, props))

    if "fukui_plus" in props and "ea" in results:
        results["fukui_plus"] = results["ea"] * HARTREE_TO_EV
    if "fukui_minus" in props and "ie" in results:
        results["fukui_minus"] = results["ie"] * HARTREE_TO_EV

    return results
