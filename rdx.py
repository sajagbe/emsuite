
# def get_redox_potential(molecule, gfec_basis_set, gfec_functional, sscf_basis_set, sscf_functional, charge, spin, method, solvent):
# #Neutral
#     rdx_n = create_molecule_object(molecule.mol.atom, basis_set=gfec_basis_set, method=method,functional=gfec_functional, charge=charge,spin=spin)
#     rdx_n_opt = optimize_and_get_equilibrium(rdx_n)
#     rdx_n = create_molecule_object(rdx_n_opt,basis_set=gfec_basis_set, method=method,functional=gfec_functional, charge=charge,spin=spin)
#     _, _, G_corr_rdx_n = get_molecule_gfec(rdx_n)

#     # #Anion
#     rdx_a, _, _ = create_charged_molecule_object(molecule.mol.atom, basis_set=gfec_basis_set, method=method, functional=gfec_functional, original_charge=charge, original_spin=spin, charge_change=-1)
#     rdx_a_opt = optimize_and_get_equilibrium(rdx_a)
#     rdx_a,_,_= create_charged_molecule_object(rdx_a_opt, basis_set=gfec_basis_set, method=method, functional=gfec_functional, original_charge=charge, original_spin=spin, charge_change=-1)
#     _,_, G_corr_rdx_a = get_molecule_gfec(rdx_a)

#     # #hbs = higher basis set solvated SCF
#     rdx_n_hbs = create_molecule_object(rdx_n.mol.atom, basis_set=basis_set, functional=functional, method=method, charge=charge, spin=spin)
#     solvated_n = solvate_molecule(rdx_n_hbs, solvent=solvent)
    
#     rdx_a_hbs,_,_ = create_charged_molecule_object(rdx_a.mol.atom, basis_set=basis_set, functional=functional, method=method, original_charge=charge, original_spin=spin, charge_change=-1)
#     solvated_a = solvate_molecule(rdx_a_hbs, solvent=solvent)

#     solvated_n.kernel()
#     solvated_a.kernel()
    
#     #GFE Adjustment
#     gfe_n = solvated_n.e_tot + G_corr_rdx_n
#     gfe_a = solvated_a.e_tot + G_corr_rdx_a

#     #Redox Gibbs Free Energy 
#     dG_red  = -1*((gfe_a - gfe_n) * HARTREE_TO_EV)
#     rdx_pot = (dG_red) - SHE_REFERENCE_POTENTIAL

#     print(f"G_corr_rdx_a: {G_corr_rdx_a}, G_corr_rdx_n: {G_corr_rdx_n}")
#     print(f"gfe_a: {gfe_a}, gfe_n: {gfe_n}, dG_red: {dG_red}, rdx_pot: {rdx_pot}")

#     return rdx_pot


# get_redox_potential(molecule, gfec_basis_set, gfec_functional, sscf_basis_set=basis_set, sscf_functional=functional, charge=charge, spin=spin, method=method, solvent=solvent)

