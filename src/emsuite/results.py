"""Typed results for surface, potential, tuning, and coupled channels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np

from emsuite.surface.io import load_surf, save_mol2, save_surf


@dataclass(frozen=True)
class _SurfGrid:
    """Shared coords/values + I/O for surf-backed channel results."""

    coords: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "coords", np.asarray(self.coords, dtype=float))
        object.__setattr__(self, "values", np.asarray(self.values, dtype=float))

    @classmethod
    def _from_surf_file(cls, path: str | Path, **extra: object) -> Self:
        coords, values = load_surf(str(path))
        return cls(coords=coords, values=values, path=str(path), **extra)  # type: ignore[arg-type]

    def _write_surf(self, path: str | Path | None, *, heterogenous: bool) -> str:
        output = str(path or self.path)
        if not output:
            raise ValueError("to_surf requires a path")
        save_surf(self.coords, self.values, output, heterogenous=heterogenous)
        return output

    def to_mol2(self, path: str | Path | None = None) -> str:
        """Write surface points as MOL2 pseudo-atoms with ``values`` in the charge column."""
        output = str(path or (Path(self.path).with_suffix(".mol2") if self.path else ""))
        if not output:
            raise ValueError("to_mol2 requires a path")
        save_mol2(self.coords, self.values, output)
        return output


@dataclass(frozen=True)
class SurfaceResult(_SurfGrid):
    path: str | None = None

    @classmethod
    def from_surf(cls, path: str | Path) -> SurfaceResult:
        return cls._from_surf_file(path)

    def to_surf(self, path: str | Path | None = None) -> str:
        return self._write_surf(path, heterogenous=False)

    def to_xyz(self, path: str | Path | None = None) -> str:
        """Write surface points as pseudo-atoms (element 'H') for viewing in a molecular viewer."""
        output = str(path or (Path(self.path).with_suffix(".xyz") if self.path else ""))
        if not output:
            raise ValueError("to_xyz requires a path")
        with open(output, "w") as f:
            f.write(f"{len(self.coords)}\n")
            f.write("Surface points\n")
            for x, y, z in self.coords:
                f.write(f"H {x:.6f} {y:.6f} {z:.6f}\n")
        return output


@dataclass(frozen=True)
class PotentialResult(_SurfGrid):
    quantity: str = "potential"
    path: str | None = None

    @classmethod
    def from_surf(cls, path: str | Path, quantity: str = "potential") -> PotentialResult:
        return cls._from_surf_file(path, quantity=quantity)

    def to_surf(self, path: str | Path | None = None) -> str:
        return self._write_surf(path, heterogenous=True)


@dataclass(frozen=True)
class TuningResult:
    results_dir: str | None = None


@dataclass(frozen=True)
class CoupledResult:
    potential: PotentialResult
    tuning: TuningResult
