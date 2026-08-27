"""Molecular geometry (XYZ)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def read_xyz(xyz_path: str | Path) -> list[tuple[str, float, float, float]]:
    lines = Path(xyz_path).read_text().strip().splitlines()
    n_atoms = int(lines[0].strip())
    atoms: list[tuple[str, float, float, float]] = []
    for line in lines[2 : 2 + n_atoms]:
        parts = line.split()
        if len(parts) < 4:
            continue
        symbol = parts[0].title() if parts[0].isalpha() else parts[0]
        atoms.append((symbol, float(parts[1]), float(parts[2]), float(parts[3])))
    return atoms


@dataclass(frozen=True)
class Geometry:
    symbols: tuple[str, ...]
    coords: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "coords", np.asarray(self.coords, dtype=float))
        if self.coords.ndim != 2 or self.coords.shape[1] != 3:
            raise ValueError("coords must have shape (n_atoms, 3)")
        if len(self.symbols) != self.coords.shape[0]:
            raise ValueError("symbols and coords length mismatch")

    @classmethod
    def from_xyz(cls, path: str | Path) -> Geometry:
        atoms = read_xyz(path)
        if not atoms:
            raise ValueError(f"No atoms found in {path}")
        symbols = tuple(symbol for symbol, _, _, _ in atoms)
        coords = np.array([[x, y, z] for _, x, y, z in atoms], dtype=float)
        return cls(symbols, coords)

    def to_xyz(self, path: str | Path, comment: str = "") -> Path:
        output = Path(path)
        lines = [str(len(self.symbols)), comment]
        for symbol, (x, y, z) in zip(self.symbols, self.coords, strict=True):
            lines.append(f"{symbol}  {x:.10f}  {y:.10f}  {z:.10f}")
        output.write_text("\n".join(lines) + "\n")
        return output
