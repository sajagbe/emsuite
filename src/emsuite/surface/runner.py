"""Surface input parsing and private runner."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from emsuite.config import parse_assignments, parse_config_file
from emsuite.config.schemas import validate_surface_params

from .generate import generate_surface

if TYPE_CHECKING:
    from emsuite.inputs import SurfaceInput


def parse_surface_input(input_file):
    """
    Parse a surface.in input file.

    Args:
        input_file (str): Path to the surface input file

    Returns:
        dict: Dictionary of parameters with defaults applied
    """
    from emsuite.inputs import SurfaceInput, _channel_defaults

    params = parse_config_file(input_file, defaults=_channel_defaults(SurfaceInput))
    parsed = parse_assignments(Path(input_file).read_text())

    if params["surface_type"].lower() == "homogenous" and "surface_charge" not in parsed:
        print("Warning: surface_charge not specified for homogenous surface, using default 0.10")

    return validate_surface_params(params)


def _run_surface(inp: SurfaceInput) -> str:
    """Execute surface generation for a validated SurfaceInput. Returns .surf path."""
    print("\n" + "=" * 60)
    print("                  Surface Generation Module")
    print("=" * 60 + "\n")

    params = inp.to_dict()

    print(f"\nInput type: {params['input_type']}")
    print(f"Input data: {params['input_data']}")
    print(f"Surface type: {params['surface_type']}")
    print(f"Output surf: {params['output_surf']}")
    if params["optimized_xyz"]:
        print(f"Optimized XYZ: {params['optimized_xyz']}")

    if params["optimize"] or (
        params["optimize"] is None and params["input_type"].upper() == "SMILES"
    ):
        print(f"Optimization: {params['optimize_method']}")
        if params["optimize_method"].lower() == "pyscf":
            print(f"  Method: {params['method']}")
            print(f"  Basis: {params['basis_set']}")
            if params["method"].lower() == "dft":
                print(f"  Functional: {params['functional']}")
            if params["solvent"]:
                print(f"  Solvent: {params['solvent']}")

    print("\n" + "-" * 60)

    output_path = generate_surface(
        input_type=params["input_type"],
        input_data=params["input_data"],
        output_surf=params["output_surf"],
        surface_density=params["surface_density"],
        surface_scale=params["surface_scale"],
        surface_type=params["surface_type"],
        surface_charge=params["surface_charge"],
        optimize=params["optimize"],
        optimize_method=params["optimize_method"],
        method=params["method"],
        basis_set=params["basis_set"],
        functional=params["functional"],
        solvent=params["solvent"],
        charge=params["charge"],
        spin=params["spin"],
        optimized_xyz=params["optimized_xyz"],
    )

    print("\n" + "-" * 60)
    print("Surface generation complete!")
    print(f"surf file: {output_path}")
    print("=" * 60 + "\n")

    return output_path
