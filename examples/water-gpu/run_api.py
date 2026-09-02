#!/usr/bin/env python3
"""Water tuning from SMILES via the Python API (surface → tuning on GPU).

Usage (on a GPU node):
    cd examples/water-gpu
    python run_api.py              # smoke: homo/lumo/gap
    python run_api.py --full       # CodexTest property set
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from emsuite import SurfaceInput, TuningInput
from emsuite.core import check_gpu_info

SMOKE_PROPS = ("homo", "lumo", "gap")
FULL_PROPS = (
    "gse",
    "homo",
    "lumo",
    "gap",
    "dm",
    "ie",
    "ea",
    "cp",
    "eng",
    "hard",
    "efl",
    "nfl",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Water SMILES → GPU tuning (Python API)")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full CodexTest property set (slower than smoke)",
    )
    args = parser.parse_args()
    properties = FULL_PROPS if args.full else SMOKE_PROPS

    workdir = Path(__file__).resolve().parent
    print(f"Working directory: {workdir}")
    print(f"GPUs detected: {check_gpu_info() or 0}")

    surf = SurfaceInput.from_config(
        input_type="SMILES",
        input_data="O",
        output_surf="Water.surf",
        optimized_xyz="Water.xyz",
        surface_density=1.0,
        surface_scale=1.0,
        surface_type="homogenous",
        surface_charge=0.10,
        optimize=True,
        optimize_method="uff",
    ).run()
    print(f"Surface: {surf.path} ({len(surf.coords)} points)")

    xyz = workdir / "Water.xyz"
    if not xyz.is_file():
        print(f"Error: expected optimized geometry at {xyz}", file=sys.stderr)
        return 1

    tuning = TuningInput.from_config(
        molecule=str(xyz),
        surface_file=surf.path,
        properties=properties,
        basis_set="6-31G*",
        method="dft",
        functional="b3lyp",
        calc_type="separate",
        parallel=True,
        num_procs=1,
        state_of_interest=1,
        triplet=False,
    ).run()

    print(f"Tuning results: {tuning.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
