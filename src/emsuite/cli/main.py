"""CLI entry point for EMSuite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from emsuite.core import print_startup_message
from emsuite.inputs import CoupledInput, PotentialInput, SurfaceInput, TuningInput


def main(argv: list[str] | None = None) -> None:
    print_startup_message()

    parser = argparse.ArgumentParser(
        prog="emsuite", description="EMSuite - Electrostatic Map Suite"
    )

    calc_type = parser.add_mutually_exclusive_group(required=True)
    calc_type.add_argument(
        "-t", "--tuning", metavar="INPUT_FILE", help="Run electrostatic tuning calculation"
    )
    calc_type.add_argument(
        "-s", "--surface", metavar="INPUT_FILE", help="Generate VDW surface from input file"
    )
    calc_type.add_argument(
        "-p",
        "--potential",
        metavar="INPUT_FILE",
        help="Compute electrostatic potential map on a surface",
    )
    calc_type.add_argument(
        "-c",
        "--coupled",
        metavar="INPUT_FILE",
        help="Run potential-derived surface charges through tuning",
    )

    args = parser.parse_args(argv)

    channels: list[tuple[Any, str | None]] = [
        (SurfaceInput, args.surface),
        (TuningInput, args.tuning),
        (PotentialInput, args.potential),
        (CoupledInput, args.coupled),
    ]
    for input_cls, path in channels:
        if path is not None:
            _run(input_cls, path)
            return


def _require_file(input_file: str) -> Path:
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found")
        sys.exit(1)
    return input_path


def _run(input_cls: type, input_file: str) -> None:
    """Build a channel Input from ``input_file`` and execute it."""
    input_cls.from_file(_require_file(input_file)).run()


if __name__ == "__main__":
    main()
