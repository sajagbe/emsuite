"""Physical constants and unit conversions via qcelemental + pint.

- **qcelemental**: QC unit conversions (hartree↔eV/kcal) and Mantina VDW radii
- **pint**: SI electromagnetic / thermal constants and length/volume scaling
"""

from __future__ import annotations

import qcelemental as qcel
from pint import UnitRegistry

_ureg = UnitRegistry()

# --- Quantum-chemistry energy conversions (qcelemental / CODATA) ---
HARTREE_TO_EV: float = float(qcel.constants.conversion_factor("hartree", "eV"))
HARTREE_TO_KCAL: float = float(qcel.constants.conversion_factor("hartree", "kcal/mol"))

# --- Length / volume (pint; matches CODATA × metric prefixes) ---
ANGSTROM_TO_METER: float = float((1 * _ureg.angstrom).to(_ureg.meter).magnitude)
A3_TO_M3: float = float((1 * _ureg.angstrom**3).to(_ureg.meter**3).magnitude)

# --- Electromagnetic / thermal (pint) ---
EPSILON_0: float = float((1 * _ureg.epsilon_0).to("F/m").magnitude)
ELEMENTARY_CHARGE: float = float((1 * _ureg.elementary_charge).to("C").magnitude)


def kT_over_e_volts(temperature_K: float = 298.0) -> float:
    """Thermal voltage kT/e in volts (APBS φ is in kT/e)."""
    return float(
        ((_ureg.boltzmann_constant * (temperature_K * _ureg.K)) / _ureg.elementary_charge)
        .to(_ureg.volt)
        .magnitude
    )


# Historical APBS convention in this package: room temperature ≈ 298 K.
KT_E_TO_V: float = kT_over_e_volts(298.0)

# Fallback VDW radius (Å) when Mantina table has no entry for an element.
_DEFAULT_VDW_RADIUS_A: float = 1.50


def vdw_radius_angstrom(symbol: str, *, missing: float = _DEFAULT_VDW_RADIUS_A) -> float:
    """Mantina (2009) van der Waals radius in Å via qcelemental."""
    return float(qcel.vdwradii.get(symbol, units="angstrom", missing=missing))
