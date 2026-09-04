import os
import subprocess
import sys

import numpy as np
from pyscf import tdscf

from ._gpu import GPU_AVAILABLE


def _mf_has_mm_charges(mf):
    """True if mf carries external MM / surface charges that must not be pickled away."""
    if getattr(mf, "_emsuite_has_mm", False):
        return True
    # CPU pyscf.qmmm wrappers (e.g. QMMMRKS)
    if "qmmm" in type(mf).__module__:
        return True
    return hasattr(mf, "mm_mol")


def create_td_molecule_object(mf, nstates=5, triplet=False, force_single_gpu=False):
    """
    Create a time-dependent (TD) calculation object for excited states.

    For multi-GPU systems, uses subprocess isolation to force single-GPU mode.
    Passes data via pickle to avoid checkpoint file corruption.

    Args:
        mf: Converged ground state SCF object
        nstates: Number of excited states to calculate
        triplet: Whether to calculate triplet states (True) or singlet (False)
        force_single_gpu: If True, skip subprocess isolation (for Ray workers)

    Returns:
        TD object with calculated excited states
    """

    # Check if multiple GPUs are visible
    current_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    visible_devices = current_cuda_devices.split(",")

    # MM + multi-GPU pickle path drops surface charges (subprocess rebuilds vacuum SCF).
    # Keep TDDFT in-process whenever MM is attached (combined / wsc / Ray separate).
    if _mf_has_mm_charges(mf) and not force_single_gpu:
        force_single_gpu = True

    # Determine if we should use subprocess isolation
    use_subprocess = (
        len(visible_devices) > 1
        and hasattr(mf, "to_cpu")
        and callable(mf.to_cpu)
        and GPU_AVAILABLE
        and not force_single_gpu
    )

    if use_subprocess:
        print("Multi-GPU detected. Using subprocess for single-GPU TDDFT...")

        # Get all necessary data from mf object
        is_dft = hasattr(mf, "xc")
        xc_functional = mf.xc if is_dft else None

        # Convert to CPU to get numpy arrays
        mf_cpu = mf.to_cpu()

        # Extract all data needed for subprocess
        import pickle
        import tempfile

        data = {
            "atom": mf_cpu.mol.atom,
            "basis": mf_cpu.mol.basis,
            "charge": mf_cpu.mol.charge,
            "spin": mf_cpu.mol.spin,
            "mo_energy": mf_cpu.mo_energy,
            "mo_coeff": mf_cpu.mo_coeff,
            "mo_occ": mf_cpu.mo_occ,
            "is_dft": is_dft,
            "xc": xc_functional,
            "nstates": nstates,
            "triplet": triplet,
        }

        # Save data to pickle
        with open("td_input.pkl", "wb") as f:
            pickle.dump(data, f)

        # Create subprocess script
        script = f"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '{visible_devices[0]}'

from pyscf import gto, dft, scf
import pickle
import numpy as np

# Load data
with open('td_input.pkl', 'rb') as f:
    data = pickle.load(f)

# Recreate molecule
mol = gto.Mole()
mol.atom = data['atom']
mol.basis = data['basis']
mol.charge = data['charge']
mol.spin = data['spin']
mol.build()

# Recreate SCF object
is_unrestricted = mol.spin > 0
if data['is_dft']:
    mf = dft.UKS(mol) if is_unrestricted else dft.RKS(mol)
    mf.xc = data['xc']
else:
    mf = scf.UHF(mol) if is_unrestricted else scf.RHF(mol)

# Convert to GPU
mf = mf.to_gpu()

# CRITICAL: Don't set chkfile to prevent checkpoint corruption
mf.chkfile = None
mf.verbose = 4

# Inject MOs as the density guess, then re-converge SCF on this single GPU.
# Multi-GPU SCF MOs injected without re-SCF often trigger:
#   RuntimeError: GGT matrix is not positive definite
# on larger chromophores (e.g. retinal) even when SCF energy looked fine.
import cupy as cp
mf.mo_energy = cp.asarray(data['mo_energy'])
mf.mo_coeff = cp.asarray(data['mo_coeff'])
mf.mo_occ = cp.asarray(data['mo_occ'])
dm0 = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
print('Re-converging SCF on single GPU before TDDFT...')
mf.kernel(dm0=dm0)
if not mf.converged:
    raise RuntimeError('Single-GPU SCF re-convergence failed before TDDFT')
print(f'Single-GPU SCF converged: E = {{float(mf.e_tot):.10f}} Ha')

# Create and run TDDFT
td = mf.TDDFT() if data['is_dft'] else mf.TDHF()
td.singlet = not data['triplet']
td.nstates = data['nstates']
td.verbose = 4
td.kernel()

# Convert results to numpy for pickling
def to_numpy(arr):
    '''Convert CuPy array to NumPy array if needed.'''
    if hasattr(arr, 'get'):
        return arr.get()
    return arr

e_np = to_numpy(td.e)
if e_np is None or len(e_np) == 0 or not np.isfinite(e_np).all():
    raise RuntimeError(f'TDDFT returned non-finite energies: {{e_np}}')

# Energies + xy are required for exe. Oscillator strengths are optional:
# gpu4pyscf can raise in td.oscillator_strength() (einsum unpack error)
# even after a successful TDDFT kernel; parent reconstructs from e/xy only.
osc = None
try:
    osc = to_numpy(td.oscillator_strength()).tolist()
except Exception as osc_err:
    print(f"WARNING: td.oscillator_strength() failed ({{osc_err}}); continuing with energies only")

results = {{
    'e': e_np.tolist(),
    'xy': [[to_numpy(xy[0]).tolist(), to_numpy(xy[1]).tolist()] for xy in td.xy],
    'oscillator_strength': osc,
}}

with open('td_results.pkl', 'wb') as f:
    pickle.dump(results, f)
"""

        # Write and run subprocess
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            script_path = f.name
            f.write(script)

        try:
            subprocess.run(
                [sys.executable, script_path],
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": visible_devices[0]},
            )

            # Load results
            with open("td_results.pkl", "rb") as f:
                results = pickle.load(f)

            # Reconstruct TD object with results
            if hasattr(mf_cpu, "TDDFT"):
                td = mf_cpu.TDDFT()
            else:
                td = tdscf.TDDFT(mf_cpu)

            # Inject results without running kernel
            td.e = np.array(results["e"])
            td.xy = [(np.array(xy[0]), np.array(xy[1])) for xy in results["xy"]]
            td.converged = True
            if td.e.size == 0 or not np.isfinite(td.e).all():
                raise RuntimeError(f"TDDFT subprocess returned non-finite energies: {td.e}")

            print(f"TDDFT completed in subprocess on GPU {visible_devices[0]}, states: {len(td.e)}")
            sys.stdout.flush()

        except subprocess.CalledProcessError as e:
            print("\n" + "=" * 70)
            print("SUBPROCESS FAILED - TDDFT Error")
            print("=" * 70)
            print(f"Return code: {e.returncode}")
            print("\n--- STDOUT ---")
            print(e.stdout if e.stdout else "(empty)")
            print("\n--- STDERR ---")
            print(e.stderr if e.stderr else "(empty)")
            print("=" * 70)

            # Try to show the script that failed
            if os.path.exists(script_path):
                print("\n--- Failed Script ---")
                with open(script_path) as f:
                    print(f.read())
                print("=" * 70)

            raise RuntimeError(f"TDDFT subprocess failed with code {e.returncode}") from e

        finally:
            # Cleanup
            if os.path.exists(script_path):
                os.unlink(script_path)
            if os.path.exists("td_input.pkl"):
                os.unlink("td_input.pkl")
            if os.path.exists("td_results.pkl"):
                os.unlink("td_results.pkl")

        return td

    else:
        # Single GPU or CPU - normal path
        print(f"Running TDDFT in current process (force_single_gpu={force_single_gpu})")
        sys.stdout.flush()

        if hasattr(mf, "with_solvent"):
            td = tdscf.TDDFT(mf) if hasattr(mf, "xc") else tdscf.TDHF(mf)
        else:
            td = mf.TDDFT() if hasattr(mf, "TDDFT") else mf.TDHF()

        td.singlet = not triplet
        td.nstates = nstates
        td.kernel()

        e = td.e
        if hasattr(e, "get"):
            e = e.get()
        e = np.asarray(e)
        if e.size == 0 or not np.isfinite(e).all():
            raise RuntimeError(f"TDDFT returned non-finite energies: {e}")

        return td


##############################################
