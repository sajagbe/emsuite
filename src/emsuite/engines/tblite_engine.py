"""TBLite-backed fast engine (GFN-xTB) for screening workflows."""

from __future__ import annotations

from pathlib import Path


class TBLiteEngine:
    """Semi-empirical xTB backend via optional ``tblite`` dependency."""

    name = "tblite"

    def __init__(self) -> None:
        self._tblite = None
        try:
            import tblite  # noqa: F401

            self._tblite = True
        except ImportError:
            self._tblite = None

    def is_available(self) -> bool:
        return self._tblite is not None

    def describe(self) -> str:
        if self.is_available():
            return "TBLite GFN-xTB backend (install with uv sync --extra mlip)."
        return "TBLite not installed — add optional dependency: uv sync --extra mlip"

    def _atoms(self, xyz_path: str, charge: int = 0):
        from ase.io import read
        from tblite.ase import TBLite

        atoms = read(xyz_path)
        atoms.calc = TBLite(method="GFN2-xTB", charge=charge)
        return atoms

    def optimize_geometry(self, xyz_path: str, charge: int = 0, **kwargs) -> str:
        if not self.is_available():
            raise ImportError(self.describe())
        from ase.io import write
        from ase.optimize import BFGS

        atoms = self._atoms(xyz_path, charge=charge)
        BFGS(atoms).run(fmax=0.05, steps=kwargs.get("max_steps", 100))
        out = Path(xyz_path).with_name(Path(xyz_path).stem + "_xtb.xyz")
        write(out, atoms)
        return str(out)

    def single_point_energy(self, xyz_path: str, charge: int = 0, **kwargs) -> float:
        if not self.is_available():
            raise ImportError(self.describe())
        atoms = self._atoms(xyz_path, charge=charge)
        return float(atoms.get_potential_energy())


# MLIPEngine delegates to TBLite when available
from .mlip_engine import MLIPEngine as _MLIPStub  # noqa: E402

__all__ = ["TBLiteEngine"]
