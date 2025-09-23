from pyscf import gto, dft, scf, lib
import time


lib.num_threads(8)

def create_charged_molecule_object(
    atom_input,
    basis_set,
    method='dft',
    functional='b3lyp',
    original_charge=0,
    charge_change=0,
    gpu=False,
    spin_guesses=None,
):
    charge = original_charge + charge_change
    spin_guesses = spin_guesses or [0, 1, 2, 3, 4]

    results = []  # store (spin, energy, mf)

    for spin in spin_guesses:
        try:
            mol = gto.Mole()
            mol.atom = atom_input
            mol.basis = basis_set
            mol.charge = charge
            mol.spin = spin   # number of unpaired electrons (2S)
            mol.build()

            # RKS for singlet, UKS for open shell
            if method.lower() == 'dft':
                mol.xc = functional
                mf = dft.UKS(mol) if spin > 0 else dft.RKS(mol)
            elif method.lower() == 'hf':
                mf = scf.UHF(mol) if spin > 0 else scf.RHF(mol)
            else:
                raise ValueError("Method must be 'dft' or 'hf'")
            if gpu:
                mf = mf.to_gpu()
            mf = mf.newton()
            energy = mf.kernel()
            if mf.converged:
                print(f"Spin {spin} (2S+1={spin+1}) converged: E = {energy:.6f} Ha")
                results.append((spin, energy, mf))
            else:
                print(f"Spin {spin} (2S+1={spin+1}) did NOT converge")

        except Exception as e:
            print(f"Spin {spin} failed: {e}")

    if results:
        # pick lowest energy among converged spins
        best_spin, best_energy, best_mf = min(results, key=lambda x: x[1])
        print(f"\nLowest energy: spin={best_spin} (2S+1={best_spin}), E={best_energy:.6f} Ha")
        return best_mf, best_spin, best_energy
    else:
        print("No spin converged for this species.")
        return None, None, None



best_mf, best_spin, best_energy = create_charged_molecule_object(
    atom_input='n-phe.xyz',
    basis_set="6-31g",
    method='dft',
    functional='b3lyp',
    original_charge=0,
    charge_change=-1,
    gpu=True,
    # spin_guesses=[0, 1]
)

print(best_spin, best_energy, best_mf.kernel() if best_mf else None)

# srun -p qGPU24 -A CHEM9C4 --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gres=gpu:1 --time=01:00:00 --mem=8G --pty ems

