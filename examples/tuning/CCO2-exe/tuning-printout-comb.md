

============================================================
                   Electrostatic Map Suite
                    By Stephen O. Ajagbe
============================================================
============================================================
                  Electrostatic Tuning Maps
             Built on efforts by the Gozem Lab
   See: https://pubs.acs.org/doi/10.1021/acs.jpcb.9b00489
============================================================



CuPy not installed - CPU mode only.
For GPU acceleration: pip install emsuite[gpu]

Calculating Tuning of:  ['exe']
Using molecular states: {'neutral': True, 'td': True}


============================================================
                Calculation Type: combined
                Number of surface points: 67
                Parallel Processing: True
============================================================


Logging results to: logs_20260325_014834
Using CPU as requested.
converged SCF energy = -155.028768605467
Spin 0 (2S+1=1) converged: E = -155.028769 Ha

Lowest energy: spin=0 (2S+1=1), E=-155.028769 Ha
Saving CPU object type: <class 'pyscf.dft.rks.RKS'>
Saving functional (from parameter): b3lyp
Saved to molecule_alone.chk, Energy: -155.028768605467 (CPU)
Running TDDFT in current process (force_single_gpu=True)
Excited State energies (eV)
[7.40202239]


============================================================
                Raw Properties (No Surface)
                Total raw properties calculated: 1
                s1_exe: 7.402026
============================================================


Running combined calculation with all 67 surface points...
Summary log initialized: logs_20260325_014834/calculation_summary.out

=== Resurrecting /home/users/sajagbe2/Desktop/packages/emsuite/venv-pypi-test/molecule_alone.chk ===
Loaded XC functional: b3lyp
Creating DFT object with xc=b3lyp


******** <class 'pyscf.dft.rks.RKS'> ********
method = RKS
initial guess = chkfile
damping factor = 0
level_shift factor = 0
DIIS = <class 'pyscf.scf.diis.CDIIS'>
diis_start_cycle = 1
diis_space = 8
diis_damp = 0
SCF conv_tol = 1e-09
SCF conv_tol_grad = None
SCF max_cycles = 50
direct_scf = True
direct_scf_tol = 1e-13
chkfile to save SCF result = /tmp/tmpwtny9g46.chk
max_memory 4000 MB (current use 241 MB)
XC library pyscf.dft.libxc version 7.0.0
    S. Lehtola, C. Steigemann, M. J.T. Oliveira, and M. A.L. Marques.,  SoftwareX 7, 1–5 (2018)
XC functionals = b3lyp
    P. J. Stephens, F. J. Devlin, C. F. Chabalowski, and M. J. Frisch.,  J. Phys. Chem. 98, 11623 (1994)
small_rho_cutoff = 1e-07
Set gradient conv threshold to 3.16228e-05
init E= -155.028768605456
  HOMO = -0.260574538961987  LUMO = 0.0672407765319143
cycle= 1 E= -155.028768605452  delta_E= 3.24e-12  |g|= 6.17e-06  |ddm|= 8.04e-06
  HOMO = -0.260572890409525  LUMO = 0.0672412713481298
Extra cycle  E= -155.028768605444  delta_E= 8.41e-12  |g|= 9.15e-06  |ddm|= 1.07e-05
converged SCF energy = -155.028768605444
Resurrected CPU DFT object: <class 'pyscf.dft.rks.RKS'>, Energy: -155.02876860544393
Running TDDFT in current process (force_single_gpu=False)


******** <class 'pyscf.tdscf.rks.TDDFT'> for <class 'pyscf.dft.rks.RKS'> ********
nstates = 1 singlet
deg_eia_thresh = 1.000e-03
wfnsym = None
conv_tol = 1e-05
eigh lindep = 1e-12
eigh level_shift = 0
eigh max_cycle = 100
chkfile = /home/users/sajagbe2/Desktop/packages/emsuite/venv-pypi-test/molecule_alone.chk
max_memory 4000 MB (current use 289 MB)


Excited State energies (eV)
[7.40200813]


******** <class 'pyscf.qmmm.itrf.QMMMRKS'> ********
method = QMMMRKS
initial guess = chkfile
damping factor = 0
level_shift factor = 0
DIIS = <class 'pyscf.scf.diis.CDIIS'>
diis_start_cycle = 1
diis_space = 8
diis_damp = 0
SCF conv_tol = 1e-09
SCF conv_tol_grad = None
SCF max_cycles = 50
direct_scf = True
direct_scf_tol = 1e-13
chkfile to save SCF result = molecule_alone.chk
max_memory 4000 MB (current use 258 MB)
XC library pyscf.dft.libxc version 7.0.0
    S. Lehtola, C. Steigemann, M. J.T. Oliveira, and M. A.L. Marques.,  SoftwareX 7, 1–5 (2018)
XC functionals = b3lyp
    P. J. Stephens, F. J. Devlin, C. F. Chabalowski, and M. J. Frisch.,  J. Phys. Chem. 98, 11623 (1994)
small_rho_cutoff = 1e-07
** Add background charges for QMMMRKS **
Set gradient conv threshold to 3.16228e-05
init E= -154.97522881075
  HOMO = -1.76972059531193  LUMO = -1.38687515814115
cycle= 1 E= -154.987292247733  delta_E= -0.0121  |g|= 0.18  |ddm|= 0.811
  HOMO = -1.71549794657579  LUMO = -1.35296134597524
cycle= 2 E= -154.985396151115  delta_E= 0.0019  |g|= 0.198  |ddm|= 0.288
  HOMO = -1.73664176638429  LUMO = -1.36381501585277
cycle= 3 E= -154.995128672427  delta_E= -0.00973  |g|= 0.0115  |ddm|= 0.156
  HOMO = -1.738464601004  LUMO = -1.36470387260189
cycle= 4 E= -154.995151265823  delta_E= -2.26e-05  |g|= 0.00377  |ddm|= 0.0105
  HOMO = -1.73787847486  LUMO = -1.3646047369046
cycle= 5 E= -154.995153858316  delta_E= -2.59e-06  |g|= 0.000578  |ddm|= 0.00316
  HOMO = -1.73788936408494  LUMO = -1.36455815078986
cycle= 6 E= -154.9951539281  delta_E= -6.98e-08  |g|= 8.46e-05  |ddm|= 0.000578
  HOMO = -1.73788751367871  LUMO = -1.36455771223067
cycle= 7 E= -154.995153929729  delta_E= -1.63e-09  |g|= 3.16e-05  |ddm|= 0.000101
  HOMO = -1.73788864764102  LUMO = -1.36455872398699
cycle= 8 E= -154.99515392997  delta_E= -2.4e-10  |g|= 3.89e-06  |ddm|= 2.69e-05
  HOMO = -1.73788841607506  LUMO = -1.36455835222842
Extra cycle  E= -154.99515392997  delta_E=    0  |g|= 3.49e-06  |ddm|= 6.43e-06
converged SCF energy = -154.99515392997
Running TDDFT in current process (force_single_gpu=False)


******** <class 'pyscf.tdscf.rks.TDDFT'> for <class 'pyscf.qmmm.itrf.QMMMRKS'> ********
nstates = 1 singlet
deg_eia_thresh = 1.000e-03
wfnsym = None
conv_tol = 1e-05
eigh lindep = 1e-12
eigh level_shift = 0
eigh max_cycle = 100
chkfile = molecule_alone.chk
max_memory 4000 MB (current use 289 MB)


Excited State energies (eV)
[8.58969659]
Combined effects: {'s1_exe_effect': np.float64(1.1876890750834352)}

Final statistics written to: logs_20260325_014834/calculation_summary.out
Raw properties appended to: logs_20260325_014834/calculation_summary.out
Created: CCO_opt2_s1_exe.mol2
Created: CCO_opt2_s1_exe_normalized.mol2

Created: CCO_opt2_tuning_summary.csv

Organizing results into: results_CCO_opt2_2026-03-25_01-49-17/
  Moved: CCO_opt2_tuning_summary.csv
  Moved 2 MOL2 files
  Added normalization parameters to summary
  Moved: logs_20260325_014834/ -> logs/

Fetching inspirational quote...


  Well, happy birthday, Jesus. Sorry that your party’s so lame. 
                     - Michael Scott

