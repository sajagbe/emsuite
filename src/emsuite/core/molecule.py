import os
import shutil
import tempfile

import numpy as np
from pyscf import dft, gto, lib, scf
from pyscf.solvent import smd

from ._gpu import GPU_AVAILABLE, cp


def create_molecule_object(
    atom_input,
    basis_set,
    method="dft",
    functional="b3lyp",
    original_charge=0,
    charge_change=0,
    gpu=True,
    spin_guesses=None,  # Note PySCF uses 2S = number of unpaired electrons not multiplicity (2S+1), so spin=0 is singlet, spin=1 is doublet, etc.
):
    """
    Create a PySCF molecule object with optimal spin configuration.

    This function creates a PySCF molecule object by testing different spin states
    and returns the one with the lowest energy. It supports both DFT and HF methods
    and can utilize GPU acceleration if available.

    Args:
        atom_input (str or list): Atomic coordinates (XYZ file path or coordinate list)
        basis_set (str): Basis set name (e.g., 'sto-3g', '6-31g*'), a list is provided in the method-info basis-sets file
        method (str, optional): Ab initio method ('dft' or 'hf'). Defaults to 'dft'.
        functional (str, optional): DFT functional name. Defaults to 'b3lyp',however, an extensive list is provided in the method-info functionals csv file with codes for easy access
                                    e.g HYB_GGA_XC_WB97X_D3 can also be used with code 399.
        original_charge (int, optional): Base molecular charge. Defaults to 0.
        charge_change (int, optional): Charge modification. Defaults to 0. Useful for generating ions.
        gpu (bool, optional): Use GPU acceleration if available. Defaults to True.
        spin_guesses (list, optional): List of spin multiplicities to test.
                                     Defaults to [0, 1, 2, 3, 4]. Uses 2S notation not multiplicity (2S+1).
                                     Important for open-shell systems.

    Returns:
        pyscf.scf object gpu4pyscf.scf object or None: The converged SCF object with lowest energy,
                                 or None if no spin state converged.

    Note:
        - Uses 2S notation (0=singlet, 1=doublet, etc.)
        - Automatically tries SOSCF if initial SCF doesn't converge
        - Prints convergence information for each spin state
    """
    charge = original_charge + charge_change
    if spin_guesses is None:
        spin_guesses = [0, 1, 2, 3, 4]
    elif isinstance(spin_guesses, int):
        spin_guesses = [spin_guesses]
    elif isinstance(spin_guesses, list) and len(spin_guesses) == 0:
        spin_guesses = [0, 1, 2, 3, 4]

    results = []  # store (spin, energy, mf)

    for spin in spin_guesses:
        try:
            mol = gto.Mole()
            mol.atom = atom_input
            mol.basis = basis_set
            mol.charge = charge
            mol.spin = spin
            # mol.verbose = 4
            mol.build()

            # RKS for singlet, UKS for open shell
            if method.lower() == "dft":
                mf = dft.UKS(mol, xc=functional) if spin > 0 else dft.RKS(mol, xc=functional)
            elif method.lower() == "hf":
                mf = scf.UHF(mol) if spin > 0 else scf.RHF(mol)
            else:
                raise ValueError("Method must be 'dft' or 'hf'")

            # Move to GPU if available and requested
            if gpu and GPU_AVAILABLE:
                mf = mf.to_gpu()
            elif gpu and not GPU_AVAILABLE:
                print("GPU requested but not available - using CPU.")
            else:
                print("Using CPU as requested.")

            energy = mf.kernel()

            # Try SOSCF if not converged
            if not mf.converged:
                mf = mf.newton()
                energy = mf.kernel()

            if mf.converged:
                print(f"Spin {spin} (2S+1={spin + 1}) converged: E = {energy:.6f} Ha")
                results.append((spin, energy, mf))
            else:
                print(f"Spin {spin} (2S+1={spin + 1}) did NOT converge")

        except Exception as e:
            print(f"Spin {spin} failed: {e}")

    if results:
        # pick lowest energy among converged spins
        best_spin, best_energy, best_mf = min(results, key=lambda x: x[1])
        print(f"\nLowest energy: spin={best_spin} (2S+1={best_spin + 1}), E={best_energy:.6f} Ha")
        return best_mf
    else:
        print("No spin converged for this species.")
        return None


def save_chkfile(mf, chkfile_name, functional=None):
    """Save a mean-field object to a checkpoint file."""
    is_gpu = hasattr(mf, "to_cpu") and callable(mf.to_cpu)

    print(f"Saving {'GPU' if is_gpu else 'CPU'} object type: {type(mf)}")

    mf.chkfile = chkfile_name

    # Save molecule structure
    lib.chkfile.save_mol(mf.mol, chkfile_name)

    # Handle CuPy arrays for GPU objects
    if is_gpu and GPU_AVAILABLE:
        mo_energy = (
            cp.asnumpy(mf.mo_energy) if isinstance(mf.mo_energy, cp.ndarray) else mf.mo_energy
        )
        mo_coeff = cp.asnumpy(mf.mo_coeff) if isinstance(mf.mo_coeff, cp.ndarray) else mf.mo_coeff
        mo_occ = cp.asnumpy(mf.mo_occ) if isinstance(mf.mo_occ, cp.ndarray) else mf.mo_occ
    else:
        mo_energy = mf.mo_energy
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ

    # Save SCF results
    lib.chkfile.save(
        chkfile_name,
        "scf",
        {
            "e_tot": float(mf.e_tot),
            "mo_energy": mo_energy,
            "mo_coeff": mo_coeff,
            "mo_occ": mo_occ,
        },
    )

    # CRITICAL: Save functional for DFT - use actual xc from object if not provided
    if functional:
        print(f"Saving functional (from parameter): {functional}")
        lib.chkfile.save(chkfile_name, "scf/xc", functional)
    elif hasattr(mf, "xc"):
        # Fallback: get XC from the object itself
        print(f"Saving functional (from mf.xc): {mf.xc}")
        lib.chkfile.save(chkfile_name, "scf/xc", mf.xc)
    else:
        print("No functional to save (HF method)")

    print(f"Saved to {chkfile_name}, Energy: {mf.e_tot} ({'GPU' if is_gpu else 'CPU'})")
    return mf


def resurrect_mol(chkfile_name):
    """Reconstruct and run a mean-field calculation from a checkpoint file."""
    print(f"\n=== Resurrecting {chkfile_name} ===")

    # Load molecule and scf data
    mol = lib.chkfile.load_mol(chkfile_name)
    scf_data = lib.chkfile.load(chkfile_name, "scf")

    # Determine method - try to load XC functional
    xc = None
    try:
        xc = lib.chkfile.load(chkfile_name, "scf/xc")
        # Handle bytes encoding
        if isinstance(xc, bytes):
            xc = xc.decode("utf-8")
        elif isinstance(xc, np.ndarray):
            xc = str(xc)
        print(f"Loaded XC functional: {xc}")
    except (KeyError, TypeError) as e:
        print(f"No XC functional found in checkpoint (this is HF): {e}")
        xc = None

    is_dft = xc is not None
    is_unrestricted = mol.spin != 0 or len(scf_data.get("mo_occ", [[]])) == 2

    # Create appropriate method object
    if is_dft:
        print(f"Creating DFT object with xc={xc}")
        mf = dft.UKS(mol) if is_unrestricted else dft.RKS(mol)
        mf.xc = xc
    else:
        print("Creating HF object (no XC functional found)")
        mf = scf.UHF(mol) if is_unrestricted else scf.RHF(mol)

    # Convert to GPU if available
    if GPU_AVAILABLE:
        try:
            print(f"Converting {type(mf)} to GPU...")
            mf = mf.to_gpu()
            print(f"Successfully converted to GPU: {type(mf)}")
        except Exception as e:
            print(f"Warning: Could not convert to GPU: {e}")
            print("Continuing with CPU")

    # CRITICAL: Use chkfile for initial guess but don't write back to it
    # Create a temporary checkpoint to avoid corruption
    temp_chk = tempfile.mktemp(suffix=".chk")
    shutil.copy2(chkfile_name, temp_chk)

    mf.chkfile = temp_chk  # Use temp file, not original
    mf.init_guess = "chkfile"
    mf.verbose = 4
    mf.kernel()

    # Clean up temp checkpoint
    if os.path.exists(temp_chk):
        try:
            os.remove(temp_chk)
        except OSError:
            pass

    # Set chkfile back to original for reference (but won't write to it)
    mf.chkfile = chkfile_name

    # Verify final object type
    is_gpu = hasattr(mf, "to_cpu") and callable(mf.to_cpu)
    method_type = "DFT" if hasattr(mf, "xc") else "HF"
    print(
        f"Resurrected {'GPU' if is_gpu else 'CPU'} {method_type} object: {type(mf)}, Energy: {mf.e_tot}"
    )

    return mf


##############################################
#        Molecular Object Manipulation       #
##############################################


def solvate_molecule(mf, solvent="water"):
    """
    Apply implicit solvation to a molecule using the Polarizable Continuum Model (PCM).

    This function adds solvation effects to an existing SCF object using the
    C-PCM method with SMD solvent parameters.

    Args:
        mf (pyscf.scf object): The molecular SCF object to solvate
        solvent (str, optional): Solvent name from SMD database. Defaults to 'water'.

    Returns:
        pyscf.scf object: The solvated SCF object

    Note:
        - Uses C-PCM method with Lebedev order 29 for cavity construction
        - Automatically tries SOSCF if initial SCF doesn't converge
        - Solvent parameters are taken from the PySCF SMD database
    """
    solvent = solvent.lower()
    mf = mf.PCM()
    mf.with_solvent.eps = smd.solvent_db[solvent][5]
    mf.with_solvent.method = "C-PCM"
    mf.with_solvent.lebedev_order = 29
    mf.kernel()
    if not mf.converged:
        print("SCF did not converge with solvent model. Trying SOSCF...")
        mf = mf.newton()
        mf.kernel()
        if not mf.converged:
            print("SOSCF also did not converge.")
        else:
            print("SOSCF converged.")
    return mf


def find_homo_lumo_and_gap(mf):
    """
    Calculate HOMO, LUMO energies and the HOMO-LUMO gap from an SCF object.

    This function analyzes the molecular orbitals to identify the highest occupied
    molecular orbital (HOMO) and lowest unoccupied molecular orbital (LUMO).

    Args:
        mf (pyscf.scf object): Converged SCF object containing molecular orbitals

    Returns:
        tuple: (HOMO energy, LUMO energy, HOMO-LUMO gap) in eV

    Note:
        - HOMO is the highest energy orbital with non-zero occupation
        - LUMO is the lowest energy orbital with zero occupation
        - Gap is calculated as LUMO - HOMO
    """
    homo = -float("inf")
    lumo = float("inf")
    for energy, occ in zip(mf.mo_energy, mf.mo_occ, strict=True):
        if occ > 0 and energy > homo:
            homo = energy
        if occ == 0 and energy < lumo:
            lumo = energy
    return homo, lumo, lumo - homo
