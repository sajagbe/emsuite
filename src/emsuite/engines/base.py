"""Calculation engine protocol (PySCF today, MLIP planned)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Engine(Protocol):
    """Protocol for QM/MLIP backends used by EMSuite channels."""

    def optimize_geometry(self, xyz_path: str, **kwargs) -> str:
        """Return path to optimized XYZ."""

    def single_point_energy(self, xyz_path: str, **kwargs) -> float:
        """Return energy in Hartree."""
