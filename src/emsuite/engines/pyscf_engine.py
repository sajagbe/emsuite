"""PySCF engine implementing the Engine protocol."""

from __future__ import annotations

from emsuite.core import (
    create_molecule_object,
    create_qmmm_molecule_object,
    create_td_molecule_object,
    optimize_molecule,
)


class PySCFEngine:
    """QM backend delegating to emsuite.core PySCF primitives."""

    name = "pyscf"

    def optimize_geometry(self, xyz_path: str, **kwargs) -> str:
        return optimize_molecule(xyz_path, **kwargs)

    def single_point_energy(self, xyz_path: str, **kwargs) -> float:
        mf = create_molecule_object(atom_input=xyz_path, **kwargs)
        if mf is None:
            raise RuntimeError(f"SCF did not converge for {xyz_path}")
        return float(mf.e_tot)

    def is_available(self) -> bool:
        return True

    def describe(self) -> str:
        return "PySCF/GPU4PySCF quantum chemistry backend (default)."


__all__ = [
    "PySCFEngine",
    "create_molecule_object",
    "create_qmmm_molecule_object",
    "create_td_molecule_object",
    "optimize_molecule",
]
