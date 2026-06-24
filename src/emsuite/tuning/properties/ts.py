"""Transition-state tuning helpers."""

from __future__ import annotations

from pathlib import Path

from emsuite.core import create_molecule_object

from ..constants import HARTREE_TO_KCAL


def ts_barrier_kcal(ts_xyz: str, reactant_mf, basis_set: str, method: str, functional: str) -> float:
    """
    Approximate activation energy: E(TS) - E(reactant).

    Uses the same level of theory as the tuning run.
    """
    if not Path(ts_xyz).exists() or reactant_mf is None:
        return 0.0
    ts_mf = create_molecule_object(
        atom_input=ts_xyz,
        basis_set=basis_set,
        method=method,
        functional=functional,
        original_charge=reactant_mf.mol.charge,
        charge_change=0,
        gpu=False,
        spin_guesses=[reactant_mf.mol.spin],
    )
    if ts_mf is None:
        return 0.0
    return (ts_mf.e_tot - reactant_mf.e_tot) * HARTREE_TO_KCAL
