"""XYZ to PQR conversion for APBS."""

from __future__ import annotations

from pathlib import Path

# van der Waals radii (Å) for common elements
VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
}


def write_pqr(
    atoms: list[tuple[str, float, float, float]],
    charges: list[float],
    output_path: str | Path,
    resname: str = "MOL",
) -> Path:
    output = Path(output_path)
    with output.open("w") as handle:
        for idx, ((symbol, x, y, z), charge) in enumerate(zip(atoms, charges, strict=True), 1):
            radius = VDW_RADII.get(symbol.upper(), 1.50)
            handle.write(
                f"ATOM  {idx:5d}  {symbol:>2s}  {resname:3s}     1"
                f"    {x:8.3f}{y:8.3f}{z:8.3f}{charge:8.4f}{radius:8.4f}\n"
            )
    return output
