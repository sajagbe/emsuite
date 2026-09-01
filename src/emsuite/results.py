"""Typed results for surface, potential, tuning, and coupled channels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from emsuite.surface.io import load_surf, save_mol2, save_surf


@dataclass(frozen=True)
class SurfaceResult:
    coords: np.ndarray
    values: np.ndarray
    path: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "coords", np.asarray(self.coords, dtype=float))
        object.__setattr__(self, "values", np.asarray(self.values, dtype=float))

    @classmethod
    def from_surf(cls, path: str | Path) -> SurfaceResult:
        coords, values = load_surf(str(path))
        return cls(coords=coords, values=values, path=str(path))

    def to_surf(self, path: str | Path | None = None) -> str:
        output = str(path or self.path)
        if not output:
            raise ValueError("to_surf requires a path")
        save_surf(self.coords, self.values, output)
        return output

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

    def to_mol2(self, path: str | Path | None = None) -> str:
        """Write surface points as MOL2 pseudo-atoms with ``values`` in the charge column."""
        output = str(path or (Path(self.path).with_suffix(".mol2") if self.path else ""))
        if not output:
            raise ValueError("to_mol2 requires a path")
        save_mol2(self.coords, self.values, output)
        return output


@dataclass(frozen=True)
class PotentialResult:
    coords: np.ndarray
    values: np.ndarray
    quantity: str
    path: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "coords", np.asarray(self.coords, dtype=float))
        object.__setattr__(self, "values", np.asarray(self.values, dtype=float))

    @classmethod
    def from_surf(cls, path: str | Path, quantity: str = "potential") -> PotentialResult:
        coords, values = load_surf(str(path))
        return cls(coords=coords, values=values, quantity=quantity, path=str(path))

    def to_surf(self, path: str | Path | None = None) -> str:
        output = str(path or self.path)
        if not output:
            raise ValueError("to_surf requires a path")
        save_surf(self.coords, self.values, output, heterogenous=True)
        return output


@dataclass(frozen=True)
class TuningResult:
    results_dir: str | None = None


@dataclass(frozen=True)
class CoupledResult:
    potential: PotentialResult
    tuning: TuningResult
