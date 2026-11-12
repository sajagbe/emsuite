"""XYZ to PQR conversion for APBS."""

from __future__ import annotations

from pathlib import Path

from emsuite.physconst import vdw_radius_angstrom


def write_pqr(
    atoms: list[tuple[str, float, float, float]],
    charges: list[float],
    output_path: str | Path,
    resname: str = "MOL",
) -> Path:
    output = Path(output_path)
    with output.open("w") as handle:
        for idx, ((symbol, x, y, z), charge) in enumerate(zip(atoms, charges, strict=True), 1):
            radius = vdw_radius_angstrom(symbol)
            handle.write(
                f"ATOM  {idx:5d}  {symbol:>2s}  {resname:3s}     1"
                f"    {x:8.3f}{y:8.3f}{z:8.3f}{charge:8.4f}{radius:8.4f}\n"
            )
    return output
