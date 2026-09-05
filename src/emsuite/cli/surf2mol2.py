"""Convert .surf files to MOL2 for visualization (PyMOL / VMD / Avogadro).

Usage (run from the folder that contains the .surf)::

    surf2mol2 coupled_smoke.surf
    surf2mol2 *.surf
    surf2mol2                 # convert every *.surf in the current directory
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from emsuite.results import PotentialResult


def _convert(surf_path: Path, out_path: Path | None = None) -> Path:
    if not surf_path.is_file():
        raise FileNotFoundError(f"not found: {surf_path}")
    if surf_path.suffix.lower() != ".surf":
        raise ValueError(f"expected a .surf file, got: {surf_path}")

    dest = out_path or surf_path.with_suffix(".mol2")
    PotentialResult.from_surf(surf_path, quantity="charge").to_mol2(dest)
    return Path(dest)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="surf2mol2",
        description="Convert EMSuite .surf files to MOL2 (charge column = surface values).",
    )
    parser.add_argument(
        "surfs",
        nargs="*",
        help=".surf file(s). If omitted, convert every *.surf in the current directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="MOL2",
        help="Output path (only valid with a single input file).",
    )
    args = parser.parse_args(argv)

    if args.output and len(args.surfs) != 1:
        parser.error("-o/--output requires exactly one input .surf file")

    paths = [Path(p) for p in args.surfs] if args.surfs else sorted(Path(".").glob("*.surf"))
    if not paths:
        print("No .surf files found.", file=sys.stderr)
        sys.exit(1)

    failed = 0
    for surf in paths:
        try:
            out = _convert(surf, Path(args.output) if args.output else None)
            print(f"{surf} -> {out}")
        except Exception as exc:
            print(f"Error converting {surf}: {exc}", file=sys.stderr)
            failed += 1

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
