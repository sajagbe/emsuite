"""CLI entry point for EMSuite."""

import argparse
import sys
from pathlib import Path

from emsuite.core import print_startup_message


def main():
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

    args = parser.parse_args()

    if args.tuning:
        run_tuning(args.tuning)
    elif args.surface:
        run_surface(args.surface)
    elif args.potential:
        run_potential(args.potential)
    elif args.coupled:
        run_coupled(args.coupled)


def _require_file(input_file: str) -> Path:
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found")
        sys.exit(1)
    return input_path


def run_surface(input_file: str):
    from emsuite.inputs import SurfaceInput

    SurfaceInput.from_file(_require_file(input_file)).run()


def run_tuning(input_file: str):
    from emsuite.inputs import TuningInput

    TuningInput.from_file(_require_file(input_file)).run()


def run_potential(input_file: str):
    from emsuite.inputs import PotentialInput

    PotentialInput.from_file(_require_file(input_file)).run()


def run_coupled(input_file: str):
    from emsuite.inputs import CoupledInput

    CoupledInput.from_file(_require_file(input_file)).run()


if __name__ == "__main__":
    main()
