"""Property registry and calculation setup."""

import numpy as np

from emsuite.core import find_homo_lumo_and_gap

from ..constants import HARTREE_TO_EV, HARTREE_TO_KCAL

PROPERTY_CONFIG = {
    "gse": {"deps": [], "calc": [], "unit": 1},
    "homo": {"deps": [], "calc": [], "unit": 1},
    "lumo": {"deps": [], "calc": [], "unit": 1},
    "gap": {"deps": ["homo", "lumo"], "calc": [], "unit": 1},
    "dm": {"deps": [], "calc": [], "unit": 1},
    "ie": {"deps": [], "calc": ["cation"], "unit": HARTREE_TO_KCAL},
    "ea": {"deps": [], "calc": ["anion"], "unit": HARTREE_TO_KCAL},
    "cp": {"deps": ["ie", "ea"], "calc": [], "unit": HARTREE_TO_KCAL},
    "eng": {"deps": ["cp"], "calc": [], "unit": HARTREE_TO_EV},
    "hard": {"deps": ["ie", "ea"], "calc": [], "unit": HARTREE_TO_EV},
    "efl": {"deps": ["cp", "hard"], "calc": [], "unit": HARTREE_TO_EV},
    "nfl": {"deps": ["efl"], "calc": [], "unit": HARTREE_TO_EV},
    "exe": {"deps": [], "calc": ["td"], "unit": 1},
    "osc": {"deps": [], "calc": ["td"], "unit": 1},
}


#####################################################
# Prepare Calculations' Dependencies on User Inputs #
#####################################################


def setup_calculation(requested_props):
    """
    Setup and resolve property dependencies and required calculations.

    This function takes a list of requested molecular properties and determines
    all the dependencies and quantum mechanical calculations needed to compute them.
    It handles the dependency tree resolution and maps properties to required
    calculations (neutral, cation, anion, TD).

    Args:
        requested_props (list): List of property names to calculate.
                               Use 'all' to calculate all available properties.

    Returns:
        tuple: (props_needed, calcs_needed) where:
            - props_needed (list): All properties needed including dependencies
            - calcs_needed (dict): Dictionary mapping calculation types to boolean
                                  (e.g., {'neutral': True, 'cation': False, ...})

    Note:
        Available properties: 'gse', 'homo', 'lumo', 'gap', 'dm', 'ie', 'ea',
        'cp', 'eng', 'hard', 'efl', 'nfl', 'exe', 'osc' and the all encompassing 'all'.

        Dependencies are automatically resolved (e.g., 'gap' requires 'homo' and 'lumo')
    """
    if "all" in requested_props:
        requested_props = list(PROPERTY_CONFIG.keys())

    # Resolve dependencies
    props_needed = set()

    def add_deps(prop):
        if prop in props_needed:
            return
        props_needed.add(prop)
        for dep in PROPERTY_CONFIG[prop]["deps"]:
            add_deps(dep)

    for prop in requested_props:
        add_deps(prop)

    # Determine required calculations
    calcs_needed = {"neutral": True}
    for prop in props_needed:
        for calc in PROPERTY_CONFIG[prop]["calc"]:
            calcs_needed[calc] = True

    return list(props_needed), calcs_needed


###############################################
# Calculate Properties from Molecular Objects #
###############################################


def calculate_all_properties(
    mf, anion_mf=None, cation_mf=None, td_obj=None, triplet=False, props_to_calc=None
):
    """
    Calculate a comprehensive set of molecular properties from quantum calculations.

    This function computes various molecular properties including energetic,
    electronic, and excited state properties from converged SCF objects.

    Args:
        mf (pyscf.scf or gpu4pyscf.scf object): Converged neutral molecule SCF object
        anion_mf (pyscf.scf or gpu4pyscf.scf object, optional): Converged anion SCF object for EA calculations
        cation_mf (pyscf.scf or gpu4pyscf.scf object, optional): Converged cation SCF object for IE calculations
        td_obj (pyscf.tdscf or gpu4pyscf.scf object, optional): TD object for excited state properties
        triplet (bool, optional): Whether to calculate triplet excited states. Defaults to False.
        props_to_calc (list, optional): List of properties to calculate

    Returns:
        dict: Dictionary containing calculated properties with units:
            - 'gse': Ground state energy (kcal/mol)
            - 'homo'/'lumo'/'gap': Orbital energies (eV)
            - 'dm': Dipole moment magnitude (Debye)
            - 'ie'/'ea': Ionization/electron affinity (kcal/mol)
            - 'cp': Chemical potential (kcal/mol)
            - 'eng': Electronegativity (eV)
            - 'hard': Chemical hardness (eV)
            - 'efl'/'nfl': Electrophilicity/nucleophilicity (eV)
            - 's1_exe'/'t1_exe': Excitation energies (eV)
            - 's1_osc'/'t1_osc': Oscillator strengths (dimensionless)

    Note:
        - Energies are converted from Hartree using conversion constants
        - Excited state properties are labeled with state prefix (s/t) and number
        - Missing SCF objects will skip dependent property calculations
    """
    results = {}

    # Handle None or empty props_to_calc
    if not props_to_calc:
        return results

    # Basic properties
    if "gse" in props_to_calc:
        results["gse"] = mf.e_tot * HARTREE_TO_KCAL

    if any(p in props_to_calc for p in ["homo", "lumo", "gap"]):
        homo, lumo, gap = [x * HARTREE_TO_EV for x in find_homo_lumo_and_gap(mf)]
        results.update(
            {
                p: v
                for p, v in zip(["homo", "lumo", "gap"], [homo, lumo, gap], strict=True)
                if p in props_to_calc
            }
        )

    if "dm" in props_to_calc:
        results["dm"] = np.linalg.norm(mf.dip_moment())

    # Charged state properties
    if "ie" in props_to_calc and cation_mf:
        results["ie"] = cation_mf.e_tot - mf.e_tot
    if "ea" in props_to_calc and anion_mf:
        results["ea"] = mf.e_tot - anion_mf.e_tot

    # Derived properties
    if "cp" in props_to_calc and all(k in results for k in ["ie", "ea"]):
        results["cp"] = -(results["ie"] + results["ea"]) / 2
    if "eng" in props_to_calc and "cp" in results:
        results["eng"] = -results["cp"]
    if "hard" in props_to_calc and all(k in results for k in ["ie", "ea"]):
        results["hard"] = (results["ie"] - results["ea"]) / 2
    if "efl" in props_to_calc and all(k in results for k in ["cp", "hard"]):
        results["efl"] = results["cp"] ** 2 / (2 * results["hard"]) if results["hard"] != 0 else 0
    if "nfl" in props_to_calc and "efl" in results:
        results["nfl"] = 1 / results["efl"] if results["efl"] != 0 else 0

    # Excited state properties - only if TD object exists and exe/osc properties are requested
    if td_obj and any(p in props_to_calc for p in ["exe", "osc"]):
        state_prefix = "t" if triplet else "s"

        if "exe" in props_to_calc:
            excitation_energies = td_obj.e * HARTREE_TO_EV
            for i, energy in enumerate(excitation_energies, 1):
                results[f"{state_prefix}{i}_exe"] = energy

        if "osc" in props_to_calc:
            oscillator_strengths = td_obj.oscillator_strength()
            for i, osc in enumerate(oscillator_strengths, 1):
                results[f"{state_prefix}{i}_osc"] = osc

    return results


