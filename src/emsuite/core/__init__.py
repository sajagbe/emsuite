"""Shared quantum chemistry primitives."""

from ._gpu import CUPY_AVAILABLE, cp
from .excited import create_td_molecule_object
from .oscillator_strength import as_numpy, oscillator_strength_cpu
from .hardware import check_cpu_info, check_gpu_info, print_office_quote, print_startup_message
from .io import extract_xyz_name, optimize_molecule
from .molecule import (
    create_molecule_object,
    find_homo_lumo_and_gap,
    resurrect_mol,
    save_chkfile,
    solvate_molecule,
)
from .qmmm import create_qmmm_molecule_object

__all__ = [
    "CUPY_AVAILABLE",
    "cp",
    "check_cpu_info",
    "check_gpu_info",
    "print_startup_message",
    "print_office_quote",
    "create_molecule_object",
    "save_chkfile",
    "resurrect_mol",
    "solvate_molecule",
    "find_homo_lumo_and_gap",
    "create_qmmm_molecule_object",
    "create_td_molecule_object",
    "as_numpy",
    "oscillator_strength_cpu",
    "extract_xyz_name",
    "optimize_molecule",
]
