from pyscf import qmmm

from ._gpu import cp


def create_qmmm_molecule_object(mf, coord_mm, q_mm, chkfile=None):
    """
    Create a QM/MM (Quantum Mechanics/Molecular Mechanics) calculation object.

    This function integrates classical point charges (MM region) with a quantum
    mechanical calculation. It handles both CPU and GPU-based SCF objects.

    Args:
        mf (pyscf.scf or gpu4pyscf.scf object): Base SCF object for the QM region
        coord_mm (numpy.ndarray): MM coordinates with shape [N, 3]
        q_mm (numpy.ndarray): MM point charges with shape [N]
        chkfile (str, optional): Checkpoint file for initial guess. Defaults to None.
                                 Typically mf's chkfile, useful for quick convergence.

    Returns:
        pyscf.scf or gpu4pyscf.scf object: New SCF object with MM charges integrated

    Note:
        - For GPU calculations, MM integration is performed on CPU then transferred
        - Automatically tries SOSCF if initial SCF doesn't converge
        - The MM charges modify both the core Hamiltonian and nuclear repulsion energy
    """
    if not hasattr(mf, "to_cpu"):
        mf_new = qmmm.mm_charge(mf, coord_mm, q_mm)
    else:
        mol = mf.mol
        # Create a new SCF object of the same type and settings
        mf_new = type(mf)(mol)
        mf_new.__dict__.update(mf.__dict__)
        # Move to CPU for MM charge integration
        temp_mf = mf_new.to_cpu()
        temp_mf_mm = qmmm.mm_charge(temp_mf, coord_mm, q_mm)
        v_mm = temp_mf_mm.get_hcore() - temp_mf.get_hcore()
        e_nuc_mm = temp_mf_mm.energy_nuc() - temp_mf.energy_nuc()
        v_mm_gpu = cp.asarray(v_mm)
        orig_get_hcore = mf_new.get_hcore
        orig_energy_nuc = mf_new.energy_nuc

        def get_hcore_with_mm(*args):
            hcore = orig_get_hcore()
            return hcore + v_mm_gpu

        def energy_nuc_with_mm(*args):
            return orig_energy_nuc() + e_nuc_mm

        mf_new.get_hcore = get_hcore_with_mm
        mf_new.energy_nuc = energy_nuc_with_mm
        mf_new.charge = mol.charge
        mf_new.spin = mol.spin
        mf_new.basis = mol.basis
        if hasattr(mf, "xc"):
            mf_new.xc = mf.xc

    if chkfile:
        mf_new.chkfile = chkfile
        mf_new.init_guess = "chkfile"
    mf_new.kernel()
    if not mf_new.converged:
        print("SCF did not converge with MM charges. Trying SOSCF...")
        mf_new = mf_new.newton()
        mf_new.kernel()
        if not mf_new.converged:
            print("SOSCF also did not converge.")
        else:
            print("SOSCF converged.")
    return mf_new


