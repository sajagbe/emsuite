"""Surface input parsing and CLI runner."""

from pathlib import Path

from emsuite.config import parse_assignments, parse_config_file
from emsuite.config.schemas import validate_surface_params

from .generate import generate_surface


def parse_surface_input(input_file):
    """
    Parse a surface.in input file.

    Args:
        input_file (str): Path to the surface input file

    Returns:
        dict: Dictionary of parameters with defaults applied
    """
    defaults = {
        "input_type": None,  # Required
        "input_data": None,  # Required
        "output_surf": "surface.surf",
        "optimized_xyz": None,  # Optional: custom name for optimized XYZ
        "surface_density": 1.0,
        "surface_scale": 1.0,
        "surface_type": "homogenous",
        "surface_charge": 0.10,
        "optimize": None,  # Auto-determined based on input_type
        "optimize_method": "mmff",
        "method": "dft",
        "basis_set": "6-31G*",
        "functional": "b3lyp",
        "solvent": None,
        "charge": 0,
        "spin": 0,
    }

    params = parse_config_file(input_file, defaults=defaults)
    parsed = parse_assignments(Path(input_file).read_text())

    if params["surface_type"].lower() == "homogenous" and "surface_charge" not in parsed:
        print("Warning: surface_charge not specified for homogenous surface, using default 0.10")

    return validate_surface_params(params)


def run_surface_calculation(input_file):
    """
    Main entry point for surface generation from input file.

    Args:
        input_file (str): Path to surface.in file

    Returns:
        str: Path to the generated surf file
    """
    print("\n" + "=" * 60)
    print("                  Surface Generation Module")
    print("=" * 60 + "\n")

    print(f"Reading input file: {input_file}")
    params = parse_surface_input(input_file)

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
