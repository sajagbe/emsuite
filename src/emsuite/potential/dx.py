"""Parse APBS OpenDX grid files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DxGrid:
    """Cell-centered (or APBS-native) scalar grid."""

    data: np.ndarray
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(n) for n in self.data.shape)


def parse_dx(path: str | Path) -> DxGrid:
    """Load an APBS OpenDX file into a ``(nx, ny, nz)`` array (x slowest, z fastest)."""
    counts: list[int] | None = None
    origin: list[float] | None = None
    deltas: list[list[float]] = []
    values: list[float] = []
    reading = False

    with Path(path).open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("object 1 class gridpositions counts"):
                counts = [int(tok) for tok in line.split()[-3:]]
            elif line.startswith("origin"):
                origin = [float(tok) for tok in line.split()[1:4]]
            elif line.startswith("delta"):
                deltas.append([float(tok) for tok in line.split()[1:4]])
            elif "data follows" in line:
                reading = True
            elif reading:
                if not line or line.startswith("attribute") or line.startswith("object"):
                    reading = False
                    continue
                for tok in line.split():
                    values.append(float(tok))

    if counts is None or origin is None or len(deltas) != 3:
        raise ValueError(f"DX file missing grid metadata: {path}")

    nx, ny, nz = counts
    expected = nx * ny * nz
    if len(values) < expected:
        raise ValueError(f"DX file {path} has {len(values)} values, expected {expected}")

    data = np.asarray(values[:expected], dtype=float).reshape(nx, ny, nz)
    spacing = (deltas[0][0], deltas[1][1], deltas[2][2])
    return DxGrid(data=data, origin=(origin[0], origin[1], origin[2]), spacing=spacing)
