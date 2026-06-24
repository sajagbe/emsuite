

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
                Calculation Type: separate
                Number of surface points: 67
                Parallel Processing: True
============================================================


Logging results to: logs_20260325_015003
Using CPU as requested.
converged SCF energy = -155.028768605468
Spin 0 (2S+1=1) converged: E = -155.028769 Ha

Lowest energy: spin=0 (2S+1=1), E=-155.028769 Ha
Saving CPU object type: <class 'pyscf.dft.rks.RKS'>
Saving functional (from parameter): b3lyp
Saved to molecule_alone.chk, Energy: -155.02876860546763 (CPU)
Running TDDFT in current process (force_single_gpu=True)
Excited State energies (eV)
[7.40202239]


============================================================
                Raw Properties (No Surface)
                Total raw properties calculated: 1
                s1_exe: 7.402026
============================================================


Summary log initialized: logs_20260325_015003/calculation_summary.out
Using 16 parallel processes on CPU
[36m(calculate_point_effect_cpu pid=3436382)[0m [Point 5] Running on CPU cores: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, PID: 3436382
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m === Resurrecting molecule_alone.chk ===
[36m(calculate_point_effect_cpu pid=3436382)[0m Loaded XC functional: b3lyp
[36m(calculate_point_effect_cpu pid=3436382)[0m Creating DFT object with xc=b3lyp
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m ******** <class 'pyscf.dft.rks.RKS'> ********
[36m(calculate_point_effect_cpu pid=3436382)[0m method = RKS
[36m(calculate_point_effect_cpu pid=3436382)[0m initial guess = chkfile
[36m(calculate_point_effect_cpu pid=3436382)[0m damping factor = 0
[36m(calculate_point_effect_cpu pid=3436382)[0m level_shift factor = 0
[36m(calculate_point_effect_cpu pid=3436382)[0m DIIS = <class 'pyscf.scf.diis.CDIIS'>
[36m(calculate_point_effect_cpu pid=3436382)[0m diis_start_cycle = 1
[36m(calculate_point_effect_cpu pid=3436382)[0m diis_space = 8
[36m(calculate_point_effect_cpu pid=3436382)[0m diis_damp = 0
[36m(calculate_point_effect_cpu pid=3436382)[0m SCF conv_tol = 1e-09
[36m(calculate_point_effect_cpu pid=3436382)[0m SCF conv_tol_grad = None
[36m(calculate_point_effect_cpu pid=3436382)[0m SCF max_cycles = 50
[36m(calculate_point_effect_cpu pid=3436382)[0m direct_scf = True
[36m(calculate_point_effect_cpu pid=3436382)[0m direct_scf_tol = 1e-13
[36m(calculate_point_effect_cpu pid=3436382)[0m chkfile to save SCF result = /tmp/tmpvo8rbwyq.chk
[36m(calculate_point_effect_cpu pid=3436382)[0m max_memory 4000 MB (current use 182 MB)
[36m(calculate_point_effect_cpu pid=3436382)[0m XC library pyscf.dft.libxc version 7.0.0
[36m(calculate_point_effect_cpu pid=3436382)[0m     S. Lehtola, C. Steigemann, M. J.T. Oliveira, and M. A.L. Marques.,  SoftwareX 7, 1–5 (2018)
[36m(calculate_point_effect_cpu pid=3436382)[0m XC functionals = b3lyp
[36m(calculate_point_effect_cpu pid=3436382)[0m     P. J. Stephens, F. J. Devlin, C. F. Chabalowski, and M. J. Frisch.,  J. Phys. Chem. 98, 11623 (1994)
[36m(calculate_point_effect_cpu pid=3436382)[0m small_rho_cutoff = 1e-07
[36m(calculate_point_effect_cpu pid=3436382)[0m Set gradient conv threshold to 3.16228e-05
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m init E= -155.028768605456
[36m(calculate_point_effect_cpu pid=3436383)[0m   HOMO = -0.260574538961987  LUMO = 0.0672407765319121
[36m(calculate_point_effect_cpu pid=3436383)[0m cycle= 1 E= -155.028768605452  delta_E= 3.92e-12  |g|= 6.17e-06  |ddm|= 8.04e-06
[36m(calculate_point_effect_cpu pid=3436381)[0m Extra cycle  E= -155.028768605444  delta_E= 8.7e-12  |g|= 9.15e-06  |ddm|= 1.07e-05
[36m(calculate_point_effect_cpu pid=3436381)[0m converged SCF energy = -155.028768605444
[36m(calculate_point_effect_cpu pid=3436376)[0m Resurrected CPU DFT object: <class 'pyscf.dft.rks.RKS'>, Energy: -155.02876860544364
[36m(calculate_point_effect_cpu pid=3436376)[0m Running TDDFT in current process (force_single_gpu=b3lyp)
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m ******** <class 'pyscf.tdscf.rks.TDDFT'> for <class 'pyscf.dft.rks.RKS'> ********
[36m(calculate_point_effect_cpu pid=3436376)[0m nstates = 1 singlet
[36m(calculate_point_effect_cpu pid=3436376)[0m deg_eia_thresh = 1.000e-03
[36m(calculate_point_effect_cpu pid=3436376)[0m wfnsym = None
[36m(calculate_point_effect_cpu pid=3436376)[0m conv_tol = 1e-05
[36m(calculate_point_effect_cpu pid=3436376)[0m eigh lindep = 1e-12
[36m(calculate_point_effect_cpu pid=3436376)[0m eigh level_shift = 0
[36m(calculate_point_effect_cpu pid=3436376)[0m eigh max_cycle = 100
[36m(calculate_point_effect_cpu pid=3436376)[0m chkfile = molecule_alone.chk
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m Excited State energies (eV)
[36m(calculate_point_effect_cpu pid=3436375)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m ******** <class 'pyscf.qmmm.itrf.QMMMRKS'> ********
[36m(calculate_point_effect_cpu pid=3436375)[0m method = QMMMRKS
[36m(calculate_point_effect_cpu pid=3436375)[0m chkfile to save SCF result = molecule_alone.chk
[36m(calculate_point_effect_cpu pid=3436375)[0m ** Add background charges for QMMMRKS **
[36m(calculate_point_effect_cpu pid=3436385)[0m [Point 10] Running on CPU cores: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, PID: 3436385[32m [repeated 15x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m === Resurrecting molecule_alone.chk ===[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m Loaded XC functional: b3lyp[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m Creating DFT object with xc=b3lyp[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m ******** <class 'pyscf.dft.rks.RKS'> ********[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m method = RKS[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m initial guess = chkfile[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m damping factor = 0[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m level_shift factor = 0[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m DIIS = <class 'pyscf.scf.diis.CDIIS'>[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m diis_start_cycle = 1[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m diis_space = 8[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m diis_damp = 0[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m SCF conv_tol = 1e-09[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m SCF conv_tol_grad = None[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m SCF max_cycles = 50[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m direct_scf = True[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m direct_scf_tol = 1e-13[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m chkfile to save SCF result = /tmp/tmpjyxf9nrw.chk[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m max_memory 4000 MB (current use 214 MB)[32m [repeated 32x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m XC library pyscf.dft.libxc version 7.0.0[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m     S. Lehtola, C. Steigemann, M. J.T. Oliveira, and M. A.L. Marques.,  SoftwareX 7, 1–5 (2018)[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m XC functionals = b3lyp[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m     P. J. Stephens, F. J. Devlin, C. F. Chabalowski, and M. J. Frisch.,  J. Phys. Chem. 98, 11623 (1994)[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m small_rho_cutoff = 1e-07[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m Set gradient conv threshold to 3.16228e-05[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436386)[0m init E= -155.028768605456[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m   HOMO = -0.260572890409526  LUMO = 0.0672412713481207[32m [repeated 31x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m cycle= 1 E= -155.028768605452  delta_E= 3.92e-12  |g|= 6.17e-06  |ddm|= 8.04e-06[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436386)[0m Extra cycle  E= -155.028768605444  delta_E= 8.7e-12  |g|= 9.15e-06  |ddm|= 1.07e-05[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m converged SCF energy = -155.028768605444[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m Resurrected CPU DFT object: <class 'pyscf.dft.rks.RKS'>, Energy: -155.02876860544364[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m Running TDDFT in current process (force_single_gpu=b3lyp)[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m ******** <class 'pyscf.tdscf.rks.TDDFT'> for <class 'pyscf.dft.rks.RKS'> ********[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m nstates = 1 singlet[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m deg_eia_thresh = 1.000e-03[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m wfnsym = None[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m conv_tol = 1e-05[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m eigh lindep = 1e-12[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m eigh level_shift = 0[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m eigh max_cycle = 100[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m chkfile = molecule_alone.chk[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436383)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m Running TDDFT in current process (force_single_gpu=False)
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m ******** <class 'pyscf.tdscf.rks.TDDFT'> for <class 'pyscf.qmmm.itrf.QMMMRKS'> ********
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m Excited State energies (eV)[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m ******** <class 'pyscf.qmmm.itrf.QMMMRKS'> ********[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m method = QMMMRKS[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m chkfile to save SCF result = molecule_alone.chk[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m ** Add background charges for QMMMRKS **[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m initial guess = chkfile[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m damping factor = 0[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m level_shift factor = 0[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m DIIS = <class 'pyscf.scf.diis.CDIIS'>[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m diis_start_cycle = 1[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m diis_space = 8[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m diis_damp = 0[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m SCF conv_tol = 1e-09[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m SCF conv_tol_grad = None[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m SCF max_cycles = 50[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m direct_scf = True[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m direct_scf_tol = 1e-13[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436379)[0m max_memory 4000 MB (current use 273 MB)[32m [repeated 14x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m XC library pyscf.dft.libxc version 7.0.0[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m     S. Lehtola, C. Steigemann, M. J.T. Oliveira, and M. A.L. Marques.,  SoftwareX 7, 1–5 (2018)[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m XC functionals = b3lyp[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m     P. J. Stephens, F. J. Devlin, C. F. Chabalowski, and M. J. Frisch.,  J. Phys. Chem. 98, 11623 (1994)[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m small_rho_cutoff = 1e-07[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m Set gradient conv threshold to 3.16228e-05[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m init E= -155.027977471296[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m   HOMO = -0.281401253874466  LUMO = 0.0490202968460035[32m [repeated 57x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m cycle= 1 E= -155.028390181302  delta_E= -0.000413  |g|= 0.0148  |ddm|= 0.115[32m [repeated 49x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436379)[0m Extra cycle  E= -155.028844842648  delta_E= -3.41e-13  |g|= 1.38e-06  |ddm|= 4.08e-06[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436379)[0m converged SCF energy = -155.028844842648[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436379)[0m nstates = 1 singlet[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436379)[0m deg_eia_thresh = 1.000e-03[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436379)[0m wfnsym = None[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436379)[0m conv_tol = 1e-05[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436379)[0m eigh lindep = 1e-12[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436379)[0m eigh level_shift = 0[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436379)[0m eigh max_cycle = 100[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436379)[0m chkfile = molecule_alone.chk[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436387)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m Running TDDFT in current process (force_single_gpu=False)[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436379)[0m ******** <class 'pyscf.tdscf.rks.TDDFT'> for <class 'pyscf.qmmm.itrf.QMMMRKS'> ********[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436386)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m Excited State energies (eV)[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m ******** <class 'pyscf.qmmm.itrf.QMMMRKS'> ********[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m method = QMMMRKS[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m chkfile to save SCF result = molecule_alone.chk[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m ** Add background charges for QMMMRKS **[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m initial guess = chkfile[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m damping factor = 0[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m level_shift factor = 0[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m DIIS = <class 'pyscf.scf.diis.CDIIS'>[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m diis_start_cycle = 1[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m diis_space = 8[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m diis_damp = 0[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m SCF conv_tol = 1e-09[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m SCF conv_tol_grad = None[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m SCF max_cycles = 50[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m direct_scf = True[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m direct_scf_tol = 1e-13[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m max_memory 4000 MB (current use 214 MB)[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m XC library pyscf.dft.libxc version 7.0.0[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m     S. Lehtola, C. Steigemann, M. J.T. Oliveira, and M. A.L. Marques.,  SoftwareX 7, 1–5 (2018)[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m XC functionals = b3lyp[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m     P. J. Stephens, F. J. Devlin, C. F. Chabalowski, and M. J. Frisch.,  J. Phys. Chem. 98, 11623 (1994)[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m small_rho_cutoff = 1e-07[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m Set gradient conv threshold to 3.16228e-05[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m init E= -155.02804440272[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436386)[0m   HOMO = -0.283992933485993  LUMO = 0.0470468101112769[32m [repeated 17x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436386)[0m cycle= 1 E= -155.0269653018  delta_E= -0.000447  |g|= 0.0247  |ddm|= 0.119[32m [repeated 13x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m Extra cycle  E= -155.028453704806  delta_E= 3.98e-13  |g|= 1.38e-06  |ddm|= 3.17e-06
[36m(calculate_point_effect_cpu pid=3436390)[0m converged SCF energy = -155.028453704806
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m nstates = 1 singlet
[36m(calculate_point_effect_cpu pid=3436390)[0m deg_eia_thresh = 1.000e-03
[36m(calculate_point_effect_cpu pid=3436390)[0m wfnsym = None
[36m(calculate_point_effect_cpu pid=3436390)[0m conv_tol = 1e-05
[36m(calculate_point_effect_cpu pid=3436390)[0m eigh lindep = 1e-12
[36m(calculate_point_effect_cpu pid=3436390)[0m eigh level_shift = 0
[36m(calculate_point_effect_cpu pid=3436390)[0m eigh max_cycle = 100
[36m(calculate_point_effect_cpu pid=3436390)[0m chkfile = molecule_alone.chk
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m Running TDDFT in current process (force_single_gpu=False)[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m ******** <class 'pyscf.tdscf.rks.TDDFT'> for <class 'pyscf.qmmm.itrf.QMMMRKS'> ********[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m Excited State energies (eV)[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m ******** <class 'pyscf.qmmm.itrf.QMMMRKS'> ********[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m method = QMMMRKS[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m chkfile to save SCF result = molecule_alone.chk[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m ** Add background charges for QMMMRKS **[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m initial guess = chkfile[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m damping factor = 0[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m level_shift factor = 0[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m DIIS = <class 'pyscf.scf.diis.CDIIS'>[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m diis_start_cycle = 1[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m diis_space = 8[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m diis_damp = 0[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m SCF conv_tol = 1e-09[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m SCF conv_tol_grad = None[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m SCF max_cycles = 50[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m direct_scf = True[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m direct_scf_tol = 1e-13[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436386)[0m max_memory 4000 MB (current use 273 MB)[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m XC library pyscf.dft.libxc version 7.0.0[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m     S. Lehtola, C. Steigemann, M. J.T. Oliveira, and M. A.L. Marques.,  SoftwareX 7, 1–5 (2018)[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m XC functionals = b3lyp[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m     P. J. Stephens, F. J. Devlin, C. F. Chabalowski, and M. J. Frisch.,  J. Phys. Chem. 98, 11623 (1994)[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m small_rho_cutoff = 1e-07[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m Set gradient conv threshold to 3.16228e-05[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m init E= -155.024800514991[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m   HOMO = -0.280291772859772  LUMO = 0.0489380149730669[32m [repeated 39x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m cycle= 4 E= -155.029011994105  delta_E= -1.5e-06  |g|= 0.000123  |ddm|= 0.00207[32m [repeated 35x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m Extra cycle  E= -155.028550006995  delta_E= -1.99e-13  |g|= 1.23e-06  |ddm|= 2.93e-06[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m converged SCF energy = -155.028550006995[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m nstates = 1 singlet[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m deg_eia_thresh = 1.000e-03[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m wfnsym = None[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m conv_tol = 1e-05[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m eigh lindep = 1e-12[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m eigh level_shift = 0[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m eigh max_cycle = 100[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m chkfile = molecule_alone.chk[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m [7.45940315]
[36m(calculate_point_effect_cpu pid=3436388)[0m Running TDDFT in current process (force_single_gpu=False)[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m ******** <class 'pyscf.tdscf.rks.TDDFT'> for <class 'pyscf.qmmm.itrf.QMMMRKS'> ********[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436383)[0m Excited State energies (eV)
[36m(calculate_point_effect_cpu pid=3436388)[0m max_memory 4000 MB (current use 248 MB)[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m   HOMO = -0.278960834803329  LUMO = 0.0497887325955153[32m [repeated 13x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m cycle= 7 E= -155.028591953624  delta_E= -8.87e-11  |g|= 2.09e-06  |ddm|= 2.3e-05[32m [repeated 13x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m Extra cycle  E= -155.028591953625  delta_E= -7.39e-13  |g|= 1.61e-06  |ddm|= 4.06e-06[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m converged SCF energy = -155.028591953625[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m nstates = 1 singlet[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m deg_eia_thresh = 1.000e-03[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m wfnsym = None[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m conv_tol = 1e-05[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m eigh lindep = 1e-12[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m eigh level_shift = 0[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m eigh max_cycle = 100[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m chkfile = molecule_alone.chk[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436389)[0m Excited State energies (eV)
[36m(calculate_point_effect_cpu pid=3436389)[0m [7.30202862]
[36m(calculate_point_effect_cpu pid=3436381)[0m [7.46652848]
[36m(calculate_point_effect_cpu pid=3436379)[0m [7.45518675]
[36m(calculate_point_effect_cpu pid=3436375)[0m [7.40790657]
[36m(calculate_point_effect_cpu pid=3436378)[0m [7.46959271]
[36m(calculate_point_effect_cpu pid=3436377)[0m [7.51412834]
[36m(calculate_point_effect_cpu pid=3436384)[0m [7.46564778]
[36m(calculate_point_effect_cpu pid=3436384)[0m Excited State energies (eV)[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436387)[0m [7.47441739]
[36m(calculate_point_effect_cpu pid=3436387)[0m Excited State energies (eV)
[36m(calculate_point_effect_cpu pid=3436380)[0m Excited State energies (eV)
[36m(calculate_point_effect_cpu pid=3436380)[0m [7.46060062]
[36m(calculate_point_effect_cpu pid=3436390)[0m [7.44700658]
[36m(calculate_point_effect_cpu pid=3436385)[0m [7.3615022]
[36m(calculate_point_effect_cpu pid=3436376)[0m [7.51456558]
[36m(calculate_point_effect_cpu pid=3436386)[0m [7.47819082]
[36m(calculate_point_effect_cpu pid=3436386)[0m Excited State energies (eV)[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m [7.44351959]
Point 1/67: SUCCESS (1/67 total)
Point 2/67: SUCCESS (2/67 total)
[36m(calculate_point_effect_cpu pid=3436382)[0m [7.44878833]
Point 3/67: SUCCESS (3/67 total)
Point 4/67: SUCCESS (4/67 total)
Point 5/67: SUCCESS (5/67 total)
Point 6/67: SUCCESS (6/67 total)
Point 7/67: SUCCESS (7/67 total)
Point 8/67: SUCCESS (8/67 total)
Point 9/67: SUCCESS (9/67 total)
Point 10/67: SUCCESS (10/67 total)
Point 11/67: SUCCESS (11/67 total)
Point 12/67: SUCCESS (12/67 total)
Point 13/67: SUCCESS (13/67 total)
Point 14/67: SUCCESS (14/67 total)
Point 15/67: SUCCESS (15/67 total)
Point 16/67: SUCCESS (16/67 total)
[36m(calculate_point_effect_cpu pid=3436376)[0m [Point 19] Running on CPU cores: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, PID: 3436376
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m === Resurrecting molecule_alone.chk ===
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m Loaded XC functional: b3lyp
[36m(calculate_point_effect_cpu pid=3436375)[0m Creating DFT object with xc=b3lyp
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m ******** <class 'pyscf.dft.rks.RKS'> ********
[36m(calculate_point_effect_cpu pid=3436387)[0m method = RKS
[36m(calculate_point_effect_cpu pid=3436387)[0m initial guess = chkfile
[36m(calculate_point_effect_cpu pid=3436387)[0m damping factor = 0
[36m(calculate_point_effect_cpu pid=3436387)[0m level_shift factor = 0
[36m(calculate_point_effect_cpu pid=3436387)[0m DIIS = <class 'pyscf.scf.diis.CDIIS'>
[36m(calculate_point_effect_cpu pid=3436387)[0m diis_start_cycle = 1
[36m(calculate_point_effect_cpu pid=3436387)[0m diis_space = 8
[36m(calculate_point_effect_cpu pid=3436387)[0m diis_damp = 0
[36m(calculate_point_effect_cpu pid=3436387)[0m SCF conv_tol = 1e-09
[36m(calculate_point_effect_cpu pid=3436387)[0m SCF conv_tol_grad = None
[36m(calculate_point_effect_cpu pid=3436387)[0m SCF max_cycles = 50
[36m(calculate_point_effect_cpu pid=3436387)[0m direct_scf = True
[36m(calculate_point_effect_cpu pid=3436387)[0m direct_scf_tol = 1e-13
[36m(calculate_point_effect_cpu pid=3436387)[0m chkfile to save SCF result = /tmp/tmpz2qlru6u.chk
[36m(calculate_point_effect_cpu pid=3436387)[0m max_memory 4000 MB (current use 220 MB)
[36m(calculate_point_effect_cpu pid=3436387)[0m XC library pyscf.dft.libxc version 7.0.0
[36m(calculate_point_effect_cpu pid=3436387)[0m     S. Lehtola, C. Steigemann, M. J.T. Oliveira, and M. A.L. Marques.,  SoftwareX 7, 1–5 (2018)
[36m(calculate_point_effect_cpu pid=3436387)[0m XC functionals = b3lyp
[36m(calculate_point_effect_cpu pid=3436387)[0m     P. J. Stephens, F. J. Devlin, C. F. Chabalowski, and M. J. Frisch.,  J. Phys. Chem. 98, 11623 (1994)
[36m(calculate_point_effect_cpu pid=3436387)[0m small_rho_cutoff = 1e-07
[36m(calculate_point_effect_cpu pid=3436387)[0m Set gradient conv threshold to 3.16228e-05
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m init E= -155.028768605456
[36m(calculate_point_effect_cpu pid=3436383)[0m   HOMO = -0.260574538961987  LUMO = 0.0672407765319121
[36m(calculate_point_effect_cpu pid=3436375)[0m cycle= 1 E= -155.028768605452  delta_E= 3.92e-12  |g|= 6.17e-06  |ddm|= 8.04e-06
[36m(calculate_point_effect_cpu pid=3436383)[0m Extra cycle  E= -155.028768605444  delta_E= 8.7e-12  |g|= 9.15e-06  |ddm|= 1.07e-05
[36m(calculate_point_effect_cpu pid=3436383)[0m converged SCF energy = -155.028768605444
[36m(calculate_point_effect_cpu pid=3436383)[0m Resurrected CPU DFT object: <class 'pyscf.dft.rks.RKS'>, Energy: -155.02876860544364
[36m(calculate_point_effect_cpu pid=3436383)[0m Running TDDFT in current process (force_single_gpu=b3lyp)
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m ******** <class 'pyscf.tdscf.rks.TDDFT'> for <class 'pyscf.dft.rks.RKS'> ********
[36m(calculate_point_effect_cpu pid=3436383)[0m nstates = 1 singlet
[36m(calculate_point_effect_cpu pid=3436383)[0m deg_eia_thresh = 1.000e-03
[36m(calculate_point_effect_cpu pid=3436383)[0m wfnsym = None
[36m(calculate_point_effect_cpu pid=3436383)[0m conv_tol = 1e-05
[36m(calculate_point_effect_cpu pid=3436383)[0m eigh lindep = 1e-12
[36m(calculate_point_effect_cpu pid=3436383)[0m eigh level_shift = 0
[36m(calculate_point_effect_cpu pid=3436383)[0m eigh max_cycle = 100
[36m(calculate_point_effect_cpu pid=3436383)[0m chkfile = molecule_alone.chk
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m ******** <class 'pyscf.qmmm.itrf.QMMMRKS'> ********
[36m(calculate_point_effect_cpu pid=3436379)[0m method = QMMMRKS
[36m(calculate_point_effect_cpu pid=3436379)[0m chkfile to save SCF result = molecule_alone.chk
[36m(calculate_point_effect_cpu pid=3436379)[0m ** Add background charges for QMMMRKS **
[36m(calculate_point_effect_cpu pid=3436379)[0m Excited State energies (eV)[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m [Point 20] Running on CPU cores: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, PID: 3436385[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m === Resurrecting molecule_alone.chk ===[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m Loaded XC functional: b3lyp[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m Creating DFT object with xc=b3lyp[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m ******** <class 'pyscf.dft.rks.RKS'> ********[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m method = RKS[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436379)[0m initial guess = chkfile[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436379)[0m damping factor = 0[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436379)[0m level_shift factor = 0[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436379)[0m DIIS = <class 'pyscf.scf.diis.CDIIS'>[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436379)[0m diis_start_cycle = 1[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436379)[0m diis_space = 8[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436379)[0m diis_damp = 0[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436379)[0m SCF conv_tol = 1e-09[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436379)[0m SCF conv_tol_grad = None[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436379)[0m SCF max_cycles = 50[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436379)[0m direct_scf = True[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436379)[0m direct_scf_tol = 1e-13[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m chkfile to save SCF result = /tmp/tmp0_qkqc0y.chk[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436379)[0m max_memory 4000 MB (current use 275 MB)[32m [repeated 32x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436379)[0m XC library pyscf.dft.libxc version 7.0.0[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436379)[0m     S. Lehtola, C. Steigemann, M. J.T. Oliveira, and M. A.L. Marques.,  SoftwareX 7, 1–5 (2018)[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436379)[0m XC functionals = b3lyp[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436379)[0m     P. J. Stephens, F. J. Devlin, C. F. Chabalowski, and M. J. Frisch.,  J. Phys. Chem. 98, 11623 (1994)[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436379)[0m small_rho_cutoff = 1e-07[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436379)[0m Set gradient conv threshold to 3.16228e-05[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m init E= -155.028768605456[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436386)[0m   HOMO = -0.260572890409526  LUMO = 0.0672412713481207[32m [repeated 31x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436386)[0m cycle= 1 E= -155.028768605452  delta_E= 3.92e-12  |g|= 6.17e-06  |ddm|= 8.04e-06[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m Extra cycle  E= -155.028768605444  delta_E= 8.7e-12  |g|= 9.15e-06  |ddm|= 1.07e-05[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m converged SCF energy = -155.028768605444[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m Resurrected CPU DFT object: <class 'pyscf.dft.rks.RKS'>, Energy: -155.02876860544364[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m Running TDDFT in current process (force_single_gpu=b3lyp)[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m ******** <class 'pyscf.tdscf.rks.TDDFT'> for <class 'pyscf.dft.rks.RKS'> ********[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m nstates = 1 singlet[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m deg_eia_thresh = 1.000e-03[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m wfnsym = None[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m conv_tol = 1e-05[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m eigh lindep = 1e-12[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m eigh level_shift = 0[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m eigh max_cycle = 100[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m chkfile = molecule_alone.chk[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436389)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m Running TDDFT in current process (force_single_gpu=False)
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m ******** <class 'pyscf.tdscf.rks.TDDFT'> for <class 'pyscf.qmmm.itrf.QMMMRKS'> ********
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m ******** <class 'pyscf.qmmm.itrf.QMMMRKS'> ********[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m method = QMMMRKS[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m chkfile to save SCF result = molecule_alone.chk[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m ** Add background charges for QMMMRKS **[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m Excited State energies (eV)[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m initial guess = chkfile[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m damping factor = 0[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m level_shift factor = 0[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m DIIS = <class 'pyscf.scf.diis.CDIIS'>[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m diis_start_cycle = 1[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m diis_space = 8[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m diis_damp = 0[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m SCF conv_tol = 1e-09[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m SCF conv_tol_grad = None[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m SCF max_cycles = 50[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m direct_scf = True[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m direct_scf_tol = 1e-13[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m max_memory 4000 MB (current use 218 MB)[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m XC library pyscf.dft.libxc version 7.0.0[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m     S. Lehtola, C. Steigemann, M. J.T. Oliveira, and M. A.L. Marques.,  SoftwareX 7, 1–5 (2018)[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m XC functionals = b3lyp[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m     P. J. Stephens, F. J. Devlin, C. F. Chabalowski, and M. J. Frisch.,  J. Phys. Chem. 98, 11623 (1994)[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m small_rho_cutoff = 1e-07[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m Set gradient conv threshold to 3.16228e-05[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m init E= -155.024353983773[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m   HOMO = -0.289370904148748  LUMO = 0.03560703855235[32m [repeated 63x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m cycle= 7 E= -155.024911240055  delta_E= -1.53e-10  |g|= 2.39e-06  |ddm|= 3.29e-05[32m [repeated 55x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m Extra cycle  E= -155.024911240056  delta_E= -8.53e-13  |g|= 1.43e-06  |ddm|= 4.87e-06[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m converged SCF energy = -155.024911240056[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m nstates = 1 singlet[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m deg_eia_thresh = 1.000e-03[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m wfnsym = None[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m conv_tol = 1e-05[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m eigh lindep = 1e-12[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m eigh level_shift = 0[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m eigh max_cycle = 100[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m chkfile = molecule_alone.chk[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m Running TDDFT in current process (force_single_gpu=False)[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m ******** <class 'pyscf.tdscf.rks.TDDFT'> for <class 'pyscf.qmmm.itrf.QMMMRKS'> ********[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m ******** <class 'pyscf.qmmm.itrf.QMMMRKS'> ********[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m method = QMMMRKS[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m chkfile to save SCF result = molecule_alone.chk[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m ** Add background charges for QMMMRKS **[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m Excited State energies (eV)[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m initial guess = chkfile[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m damping factor = 0[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m level_shift factor = 0[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m DIIS = <class 'pyscf.scf.diis.CDIIS'>[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m diis_start_cycle = 1[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m diis_space = 8[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m diis_damp = 0[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m SCF conv_tol = 1e-09[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m SCF conv_tol_grad = None[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m SCF max_cycles = 50[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m direct_scf = True[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m direct_scf_tol = 1e-13[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m max_memory 4000 MB (current use 216 MB)[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m XC library pyscf.dft.libxc version 7.0.0[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m     S. Lehtola, C. Steigemann, M. J.T. Oliveira, and M. A.L. Marques.,  SoftwareX 7, 1–5 (2018)[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m XC functionals = b3lyp[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m     P. J. Stephens, F. J. Devlin, C. F. Chabalowski, and M. J. Frisch.,  J. Phys. Chem. 98, 11623 (1994)[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m small_rho_cutoff = 1e-07[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m Set gradient conv threshold to 3.16228e-05[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436387)[0m init E= -155.026836968336[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436387)[0m   HOMO = -0.280721798836216  LUMO = 0.0493632437224756[32m [repeated 10x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m cycle= 4 E= -155.0278905088  delta_E= -3.76e-06  |g|= 0.000518  |ddm|= 0.00372[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436386)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m Extra cycle  E= -155.027890568143  delta_E= -2.84e-13  |g|= 1.16e-06  |ddm|= 3.29e-06
[36m(calculate_point_effect_cpu pid=3436388)[0m converged SCF energy = -155.027890568143
[36m(calculate_point_effect_cpu pid=3436388)[0m Running TDDFT in current process (force_single_gpu=False)
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m ******** <class 'pyscf.tdscf.rks.TDDFT'> for <class 'pyscf.qmmm.itrf.QMMMRKS'> ********
[36m(calculate_point_effect_cpu pid=3436388)[0m nstates = 1 singlet
[36m(calculate_point_effect_cpu pid=3436388)[0m deg_eia_thresh = 1.000e-03
[36m(calculate_point_effect_cpu pid=3436388)[0m wfnsym = None
[36m(calculate_point_effect_cpu pid=3436388)[0m conv_tol = 1e-05
[36m(calculate_point_effect_cpu pid=3436388)[0m eigh lindep = 1e-12
[36m(calculate_point_effect_cpu pid=3436388)[0m eigh level_shift = 0
[36m(calculate_point_effect_cpu pid=3436388)[0m eigh max_cycle = 100
[36m(calculate_point_effect_cpu pid=3436388)[0m chkfile = molecule_alone.chk
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m ******** <class 'pyscf.qmmm.itrf.QMMMRKS'> ********[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m method = QMMMRKS[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m chkfile to save SCF result = molecule_alone.chk[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m ** Add background charges for QMMMRKS **[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m Excited State energies (eV)[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m initial guess = chkfile[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m damping factor = 0[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m level_shift factor = 0[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m DIIS = <class 'pyscf.scf.diis.CDIIS'>[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m diis_start_cycle = 1[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m diis_space = 8[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m diis_damp = 0[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m SCF conv_tol = 1e-09[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m SCF conv_tol_grad = None[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m SCF max_cycles = 50[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m direct_scf = True[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m direct_scf_tol = 1e-13[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436387)[0m max_memory 4000 MB (current use 251 MB)[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m XC library pyscf.dft.libxc version 7.0.0[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m     S. Lehtola, C. Steigemann, M. J.T. Oliveira, and M. A.L. Marques.,  SoftwareX 7, 1–5 (2018)[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m XC functionals = b3lyp[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m     P. J. Stephens, F. J. Devlin, C. F. Chabalowski, and M. J. Frisch.,  J. Phys. Chem. 98, 11623 (1994)[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m small_rho_cutoff = 1e-07[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m Set gradient conv threshold to 3.16228e-05[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m init E= -155.028219532984[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m   HOMO = -0.283037724658694  LUMO = 0.0481073112145752[32m [repeated 50x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m cycle= 4 E= -155.028760457932  delta_E= -1.39e-06  |g|= 0.000218  |ddm|= 0.00215[32m [repeated 47x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m Extra cycle  E= -155.028760469597  delta_E= -5.68e-14  |g|= 2.41e-06  |ddm|= 5.95e-06[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m converged SCF energy = -155.028760469597[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m Running TDDFT in current process (force_single_gpu=False)[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m ******** <class 'pyscf.tdscf.rks.TDDFT'> for <class 'pyscf.qmmm.itrf.QMMMRKS'> ********[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m nstates = 1 singlet[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m deg_eia_thresh = 1.000e-03[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m wfnsym = None[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m conv_tol = 1e-05[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m eigh lindep = 1e-12[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m eigh level_shift = 0[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m eigh max_cycle = 100[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m chkfile = molecule_alone.chk[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436389)[0m [7.49833124]
[36m(calculate_point_effect_cpu pid=3436389)[0m Excited State energies (eV)
[36m(calculate_point_effect_cpu pid=3436390)[0m max_memory 4000 MB (current use 277 MB)[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m   HOMO = -0.283026343630023  LUMO = 0.0481111837482588[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m cycle= 7 E= -155.028760469597  delta_E= -2.18e-10  |g|= 2.91e-06  |ddm|= 2.86e-05[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436379)[0m Excited State energies (eV)
[36m(calculate_point_effect_cpu pid=3436379)[0m [7.48930658]
[36m(calculate_point_effect_cpu pid=3436378)[0m [7.54557912]
[36m(calculate_point_effect_cpu pid=3436383)[0m [7.36468788]
[36m(calculate_point_effect_cpu pid=3436381)[0m [7.52656471]
[36m(calculate_point_effect_cpu pid=3436384)[0m [7.30418687]
[36m(calculate_point_effect_cpu pid=3436375)[0m [7.53809272]
[36m(calculate_point_effect_cpu pid=3436377)[0m [7.48336652]
[36m(calculate_point_effect_cpu pid=3436388)[0m [7.41325131]
[36m(calculate_point_effect_cpu pid=3436388)[0m Excited State energies (eV)[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m [7.53632831]
[36m(calculate_point_effect_cpu pid=3436390)[0m [7.4938373]
[36m(calculate_point_effect_cpu pid=3436387)[0m [7.47844732]
[36m(calculate_point_effect_cpu pid=3436376)[0m [7.50481545]
[36m(calculate_point_effect_cpu pid=3436385)[0m [7.57235878]
[36m(calculate_point_effect_cpu pid=3436382)[0m [7.41426972]
[36m(calculate_point_effect_cpu pid=3436382)[0m Excited State energies (eV)[32m [repeated 6x across cluster][0m
Point 17/67: SUCCESS (17/67 total)
Point 18/67: SUCCESS (18/67 total)
[36m(calculate_point_effect_cpu pid=3436386)[0m [7.50608126]
Point 19/67: SUCCESS (19/67 total)
Point 20/67: SUCCESS (20/67 total)
Point 21/67: SUCCESS (21/67 total)
Point 22/67: SUCCESS (22/67 total)
Point 23/67: SUCCESS (23/67 total)
Point 24/67: SUCCESS (24/67 total)
Point 25/67: SUCCESS (25/67 total)
Point 26/67: SUCCESS (26/67 total)
Point 27/67: SUCCESS (27/67 total)
Point 28/67: SUCCESS (28/67 total)
Point 29/67: SUCCESS (29/67 total)
Point 30/67: SUCCESS (30/67 total)
Point 31/67: SUCCESS (31/67 total)
Point 32/67: SUCCESS (32/67 total)
[36m(calculate_point_effect_cpu pid=3436376)[0m [Point 35] Running on CPU cores: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, PID: 3436376
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m === Resurrecting molecule_alone.chk ===
[36m(calculate_point_effect_cpu pid=3436376)[0m Loaded XC functional: b3lyp
[36m(calculate_point_effect_cpu pid=3436376)[0m Creating DFT object with xc=b3lyp
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m ******** <class 'pyscf.dft.rks.RKS'> ********
[36m(calculate_point_effect_cpu pid=3436376)[0m method = RKS
[36m(calculate_point_effect_cpu pid=3436376)[0m initial guess = chkfile
[36m(calculate_point_effect_cpu pid=3436376)[0m damping factor = 0
[36m(calculate_point_effect_cpu pid=3436376)[0m level_shift factor = 0
[36m(calculate_point_effect_cpu pid=3436376)[0m DIIS = <class 'pyscf.scf.diis.CDIIS'>
[36m(calculate_point_effect_cpu pid=3436376)[0m diis_start_cycle = 1
[36m(calculate_point_effect_cpu pid=3436376)[0m diis_space = 8
[36m(calculate_point_effect_cpu pid=3436376)[0m diis_damp = 0
[36m(calculate_point_effect_cpu pid=3436376)[0m SCF conv_tol = 1e-09
[36m(calculate_point_effect_cpu pid=3436376)[0m SCF conv_tol_grad = None
[36m(calculate_point_effect_cpu pid=3436376)[0m SCF max_cycles = 50
[36m(calculate_point_effect_cpu pid=3436376)[0m direct_scf = True
[36m(calculate_point_effect_cpu pid=3436376)[0m direct_scf_tol = 1e-13
[36m(calculate_point_effect_cpu pid=3436376)[0m chkfile to save SCF result = /tmp/tmpnjq4gfd6.chk
[36m(calculate_point_effect_cpu pid=3436376)[0m max_memory 4000 MB (current use 278 MB)
[36m(calculate_point_effect_cpu pid=3436376)[0m XC library pyscf.dft.libxc version 7.0.0
[36m(calculate_point_effect_cpu pid=3436376)[0m     S. Lehtola, C. Steigemann, M. J.T. Oliveira, and M. A.L. Marques.,  SoftwareX 7, 1–5 (2018)
[36m(calculate_point_effect_cpu pid=3436376)[0m XC functionals = b3lyp
[36m(calculate_point_effect_cpu pid=3436376)[0m     P. J. Stephens, F. J. Devlin, C. F. Chabalowski, and M. J. Frisch.,  J. Phys. Chem. 98, 11623 (1994)
[36m(calculate_point_effect_cpu pid=3436376)[0m small_rho_cutoff = 1e-07
[36m(calculate_point_effect_cpu pid=3436376)[0m Set gradient conv threshold to 3.16228e-05
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m chkfile to save SCF result = /tmp/tmpkcwz_osa.chk
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m chkfile to save SCF result = /tmp/tmpsvecgimz.chk
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m init E= -155.028768605456
[36m(calculate_point_effect_cpu pid=3436384)[0m   HOMO = -0.260574538961987  LUMO = 0.0672407765319121
[36m(calculate_point_effect_cpu pid=3436375)[0m cycle= 1 E= -155.028768605452  delta_E= 3.92e-12  |g|= 6.17e-06  |ddm|= 8.04e-06
[36m(calculate_point_effect_cpu pid=3436386)[0m Excited State energies (eV)
[36m(calculate_point_effect_cpu pid=3436384)[0m Extra cycle  E= -155.028768605444  delta_E= 8.7e-12  |g|= 9.15e-06  |ddm|= 1.07e-05
[36m(calculate_point_effect_cpu pid=3436384)[0m converged SCF energy = -155.028768605444
[36m(calculate_point_effect_cpu pid=3436384)[0m Resurrected CPU DFT object: <class 'pyscf.dft.rks.RKS'>, Energy: -155.02876860544364
[36m(calculate_point_effect_cpu pid=3436384)[0m Running TDDFT in current process (force_single_gpu=b3lyp)
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m ******** <class 'pyscf.tdscf.rks.TDDFT'> for <class 'pyscf.dft.rks.RKS'> ********
[36m(calculate_point_effect_cpu pid=3436384)[0m nstates = 1 singlet
[36m(calculate_point_effect_cpu pid=3436384)[0m deg_eia_thresh = 1.000e-03
[36m(calculate_point_effect_cpu pid=3436384)[0m wfnsym = None
[36m(calculate_point_effect_cpu pid=3436384)[0m conv_tol = 1e-05
[36m(calculate_point_effect_cpu pid=3436384)[0m eigh lindep = 1e-12
[36m(calculate_point_effect_cpu pid=3436384)[0m eigh level_shift = 0
[36m(calculate_point_effect_cpu pid=3436384)[0m eigh max_cycle = 100
[36m(calculate_point_effect_cpu pid=3436384)[0m chkfile = molecule_alone.chk
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m Excited State energies (eV)
[36m(calculate_point_effect_cpu pid=3436381)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m ******** <class 'pyscf.qmmm.itrf.QMMMRKS'> ********
[36m(calculate_point_effect_cpu pid=3436381)[0m method = QMMMRKS
[36m(calculate_point_effect_cpu pid=3436381)[0m chkfile to save SCF result = molecule_alone.chk
[36m(calculate_point_effect_cpu pid=3436381)[0m ** Add background charges for QMMMRKS **
[36m(calculate_point_effect_cpu pid=3436385)[0m [Point 34] Running on CPU cores: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, PID: 3436385[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m === Resurrecting molecule_alone.chk ===[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m Loaded XC functional: b3lyp[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m Creating DFT object with xc=b3lyp[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m ******** <class 'pyscf.dft.rks.RKS'> ********[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m method = RKS[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m initial guess = chkfile[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m damping factor = 0[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m level_shift factor = 0[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m DIIS = <class 'pyscf.scf.diis.CDIIS'>[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m diis_start_cycle = 1[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m diis_space = 8[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m diis_damp = 0[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m SCF conv_tol = 1e-09[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m SCF conv_tol_grad = None[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m SCF max_cycles = 50[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m direct_scf = True[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m direct_scf_tol = 1e-13[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m chkfile to save SCF result = /tmp/tmpuq0gj94b.chk[32m [repeated 13x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m max_memory 4000 MB (current use 275 MB)[32m [repeated 32x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m XC library pyscf.dft.libxc version 7.0.0[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m     S. Lehtola, C. Steigemann, M. J.T. Oliveira, and M. A.L. Marques.,  SoftwareX 7, 1–5 (2018)[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m XC functionals = b3lyp[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m     P. J. Stephens, F. J. Devlin, C. F. Chabalowski, and M. J. Frisch.,  J. Phys. Chem. 98, 11623 (1994)[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m small_rho_cutoff = 1e-07[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m Set gradient conv threshold to 3.16228e-05[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m init E= -155.028768605456[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m   HOMO = -0.260572890409526  LUMO = 0.0672412713481207[32m [repeated 31x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m cycle= 1 E= -155.028768605452  delta_E= 3.92e-12  |g|= 6.17e-06  |ddm|= 8.04e-06[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m Extra cycle  E= -155.028768605444  delta_E= 8.7e-12  |g|= 9.15e-06  |ddm|= 1.07e-05[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m converged SCF energy = -155.028768605444[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m Resurrected CPU DFT object: <class 'pyscf.dft.rks.RKS'>, Energy: -155.02876860544364[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m Running TDDFT in current process (force_single_gpu=b3lyp)[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m ******** <class 'pyscf.tdscf.rks.TDDFT'> for <class 'pyscf.dft.rks.RKS'> ********[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m nstates = 1 singlet[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m deg_eia_thresh = 1.000e-03[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m wfnsym = None[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m conv_tol = 1e-05[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m eigh lindep = 1e-12[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m eigh level_shift = 0[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m eigh max_cycle = 100[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m chkfile = molecule_alone.chk[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m Running TDDFT in current process (force_single_gpu=False)
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m ******** <class 'pyscf.tdscf.rks.TDDFT'> for <class 'pyscf.qmmm.itrf.QMMMRKS'> ********
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m Excited State energies (eV)[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m ******** <class 'pyscf.qmmm.itrf.QMMMRKS'> ********[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m method = QMMMRKS[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m chkfile to save SCF result = molecule_alone.chk[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m ** Add background charges for QMMMRKS **[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m initial guess = chkfile[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m damping factor = 0[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m level_shift factor = 0[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m DIIS = <class 'pyscf.scf.diis.CDIIS'>[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m diis_start_cycle = 1[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m diis_space = 8[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m diis_damp = 0[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m SCF conv_tol = 1e-09[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m SCF conv_tol_grad = None[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m SCF max_cycles = 50[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m direct_scf = True[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m direct_scf_tol = 1e-13[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436378)[0m max_memory 4000 MB (current use 276 MB)[32m [repeated 12x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m XC library pyscf.dft.libxc version 7.0.0[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m     S. Lehtola, C. Steigemann, M. J.T. Oliveira, and M. A.L. Marques.,  SoftwareX 7, 1–5 (2018)[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m XC functionals = b3lyp[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m     P. J. Stephens, F. J. Devlin, C. F. Chabalowski, and M. J. Frisch.,  J. Phys. Chem. 98, 11623 (1994)[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m small_rho_cutoff = 1e-07[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m Set gradient conv threshold to 3.16228e-05[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436375)[0m init E= -155.027826572711[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436389)[0m   HOMO = -0.283296309991221  LUMO = 0.0334959114891454[32m [repeated 60x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436389)[0m cycle= 7 E= -155.023762748005  delta_E= -9.21e-11  |g|= 2.7e-06  |ddm|= 2.7e-05[32m [repeated 52x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436378)[0m Extra cycle  E= -155.024997747776  delta_E= -1.48e-12  |g|= 1.22e-06  |ddm|= 4.69e-06[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436378)[0m converged SCF energy = -155.024997747776[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436378)[0m nstates = 1 singlet[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436378)[0m deg_eia_thresh = 1.000e-03[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436378)[0m wfnsym = None[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436378)[0m conv_tol = 1e-05[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436378)[0m eigh lindep = 1e-12[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436378)[0m eigh level_shift = 0[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436378)[0m eigh max_cycle = 100[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436378)[0m chkfile = molecule_alone.chk[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m Running TDDFT in current process (force_single_gpu=False)[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m ******** <class 'pyscf.tdscf.rks.TDDFT'> for <class 'pyscf.qmmm.itrf.QMMMRKS'> ********[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436386)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m Excited State energies (eV)[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436387)[0m ******** <class 'pyscf.qmmm.itrf.QMMMRKS'> ********[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436387)[0m method = QMMMRKS[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436387)[0m chkfile to save SCF result = molecule_alone.chk[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436387)[0m ** Add background charges for QMMMRKS **[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436387)[0m initial guess = chkfile[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436387)[0m damping factor = 0[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436387)[0m level_shift factor = 0[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436387)[0m DIIS = <class 'pyscf.scf.diis.CDIIS'>[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436387)[0m diis_start_cycle = 1[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436387)[0m diis_space = 8[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436387)[0m diis_damp = 0[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436387)[0m SCF conv_tol = 1e-09[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436387)[0m SCF conv_tol_grad = None[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436387)[0m SCF max_cycles = 50[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436387)[0m direct_scf = True[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436387)[0m direct_scf_tol = 1e-13[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436387)[0m max_memory 4000 MB (current use 220 MB)[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436387)[0m XC library pyscf.dft.libxc version 7.0.0[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436387)[0m     S. Lehtola, C. Steigemann, M. J.T. Oliveira, and M. A.L. Marques.,  SoftwareX 7, 1–5 (2018)[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436387)[0m XC functionals = b3lyp[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436387)[0m     P. J. Stephens, F. J. Devlin, C. F. Chabalowski, and M. J. Frisch.,  J. Phys. Chem. 98, 11623 (1994)[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436387)[0m small_rho_cutoff = 1e-07[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436387)[0m Set gradient conv threshold to 3.16228e-05[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m   HOMO = -0.280237145022197  LUMO = 0.0496761056334986[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m cycle= 7 E= -155.027383728762  delta_E= -9.4e-11  |g|= 1.59e-06  |ddm|= 1.77e-05[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m Extra cycle  E= -155.027383728762  delta_E= -2.84e-13  |g|= 7.75e-07  |ddm|= 2.89e-06[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m converged SCF energy = -155.027383728762[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m nstates = 1 singlet[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m deg_eia_thresh = 1.000e-03[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m wfnsym = None[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m conv_tol = 1e-05[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m eigh lindep = 1e-12[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m eigh level_shift = 0[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m eigh max_cycle = 100[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m chkfile = molecule_alone.chk[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436386)[0m init E= -155.031494339917
[36m(calculate_point_effect_cpu pid=3436376)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m Running TDDFT in current process (force_single_gpu=False)
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m ******** <class 'pyscf.tdscf.rks.TDDFT'> for <class 'pyscf.qmmm.itrf.QMMMRKS'> ********
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m Excited State energies (eV)[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m ******** <class 'pyscf.qmmm.itrf.QMMMRKS'> ********[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m method = QMMMRKS[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m chkfile to save SCF result = molecule_alone.chk[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m ** Add background charges for QMMMRKS **[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m initial guess = chkfile[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m damping factor = 0[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m level_shift factor = 0[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m DIIS = <class 'pyscf.scf.diis.CDIIS'>[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m diis_start_cycle = 1[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m diis_space = 8[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m diis_damp = 0[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m SCF conv_tol = 1e-09[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m SCF conv_tol_grad = None[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m SCF max_cycles = 50[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m direct_scf = True[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m direct_scf_tol = 1e-13[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m max_memory 4000 MB (current use 218 MB)[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m XC library pyscf.dft.libxc version 7.0.0[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m     S. Lehtola, C. Steigemann, M. J.T. Oliveira, and M. A.L. Marques.,  SoftwareX 7, 1–5 (2018)[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m XC functionals = b3lyp[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m     P. J. Stephens, F. J. Devlin, C. F. Chabalowski, and M. J. Frisch.,  J. Phys. Chem. 98, 11623 (1994)[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m small_rho_cutoff = 1e-07[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m Set gradient conv threshold to 3.16228e-05[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m   HOMO = -0.279550357701546  LUMO = 0.045087011062878[32m [repeated 23x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m cycle= 2 E= -155.026840531071  delta_E= -3.58e-05  |g|= 0.00863  |ddm|= 0.0291[32m [repeated 17x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436387)[0m Extra cycle  E= -155.028245594349  delta_E= -1.14e-12  |g|= 1.68e-06  |ddm|= 4.36e-06
[36m(calculate_point_effect_cpu pid=3436387)[0m converged SCF energy = -155.028245594349
[36m(calculate_point_effect_cpu pid=3436387)[0m nstates = 1 singlet
[36m(calculate_point_effect_cpu pid=3436387)[0m deg_eia_thresh = 1.000e-03
[36m(calculate_point_effect_cpu pid=3436387)[0m wfnsym = None
[36m(calculate_point_effect_cpu pid=3436387)[0m conv_tol = 1e-05
[36m(calculate_point_effect_cpu pid=3436387)[0m eigh lindep = 1e-12
[36m(calculate_point_effect_cpu pid=3436387)[0m eigh level_shift = 0
[36m(calculate_point_effect_cpu pid=3436387)[0m eigh max_cycle = 100
[36m(calculate_point_effect_cpu pid=3436387)[0m chkfile = molecule_alone.chk
[36m(calculate_point_effect_cpu pid=3436382)[0m init E= -155.035576629377[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m Extra cycle  E= -155.031956514222  delta_E= -8.53e-13  |g|= 1.79e-06  |ddm|= 5.63e-06
[36m(calculate_point_effect_cpu pid=3436386)[0m converged SCF energy = -155.031956514222
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m nstates = 1 singlet
[36m(calculate_point_effect_cpu pid=3436386)[0m deg_eia_thresh = 1.000e-03
[36m(calculate_point_effect_cpu pid=3436386)[0m wfnsym = None
[36m(calculate_point_effect_cpu pid=3436386)[0m conv_tol = 1e-05
[36m(calculate_point_effect_cpu pid=3436386)[0m eigh lindep = 1e-12
[36m(calculate_point_effect_cpu pid=3436386)[0m eigh level_shift = 0
[36m(calculate_point_effect_cpu pid=3436386)[0m eigh max_cycle = 100
[36m(calculate_point_effect_cpu pid=3436386)[0m chkfile = molecule_alone.chk
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m Running TDDFT in current process (force_single_gpu=False)[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m ******** <class 'pyscf.tdscf.rks.TDDFT'> for <class 'pyscf.qmmm.itrf.QMMMRKS'> ********[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m Excited State energies (eV)
[36m(calculate_point_effect_cpu pid=3436385)[0m ******** <class 'pyscf.qmmm.itrf.QMMMRKS'> ********
[36m(calculate_point_effect_cpu pid=3436385)[0m method = QMMMRKS
[36m(calculate_point_effect_cpu pid=3436385)[0m chkfile to save SCF result = molecule_alone.chk
[36m(calculate_point_effect_cpu pid=3436385)[0m ** Add background charges for QMMMRKS **
[36m(calculate_point_effect_cpu pid=3436385)[0m initial guess = chkfile
[36m(calculate_point_effect_cpu pid=3436385)[0m damping factor = 0
[36m(calculate_point_effect_cpu pid=3436385)[0m level_shift factor = 0
[36m(calculate_point_effect_cpu pid=3436385)[0m DIIS = <class 'pyscf.scf.diis.CDIIS'>
[36m(calculate_point_effect_cpu pid=3436385)[0m diis_start_cycle = 1
[36m(calculate_point_effect_cpu pid=3436385)[0m diis_space = 8
[36m(calculate_point_effect_cpu pid=3436385)[0m diis_damp = 0
[36m(calculate_point_effect_cpu pid=3436385)[0m SCF conv_tol = 1e-09
[36m(calculate_point_effect_cpu pid=3436385)[0m SCF conv_tol_grad = None
[36m(calculate_point_effect_cpu pid=3436385)[0m SCF max_cycles = 50
[36m(calculate_point_effect_cpu pid=3436385)[0m direct_scf = True
[36m(calculate_point_effect_cpu pid=3436385)[0m direct_scf_tol = 1e-13
[36m(calculate_point_effect_cpu pid=3436382)[0m max_memory 4000 MB (current use 275 MB)[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m XC library pyscf.dft.libxc version 7.0.0
[36m(calculate_point_effect_cpu pid=3436385)[0m     S. Lehtola, C. Steigemann, M. J.T. Oliveira, and M. A.L. Marques.,  SoftwareX 7, 1–5 (2018)
[36m(calculate_point_effect_cpu pid=3436385)[0m XC functionals = b3lyp
[36m(calculate_point_effect_cpu pid=3436385)[0m     P. J. Stephens, F. J. Devlin, C. F. Chabalowski, and M. J. Frisch.,  J. Phys. Chem. 98, 11623 (1994)
[36m(calculate_point_effect_cpu pid=3436385)[0m small_rho_cutoff = 1e-07
[36m(calculate_point_effect_cpu pid=3436385)[0m Set gradient conv threshold to 3.16228e-05
[36m(calculate_point_effect_cpu pid=3436380)[0m   HOMO = -0.277566709006587  LUMO = 0.0496225544870926[32m [repeated 39x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m cycle= 7 E= -155.027564209216  delta_E= -9.12e-11  |g|= 2.18e-06  |ddm|= 2.09e-05[32m [repeated 37x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m init E= -155.028004310128[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m Extra cycle  E= -155.028451774928  delta_E= -2.5e-12  |g|= 2.15e-06  |ddm|= 6.96e-06[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m converged SCF energy = -155.027564209216[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m nstates = 1 singlet[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m deg_eia_thresh = 1.000e-03[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m wfnsym = None[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m conv_tol = 1e-05[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m eigh lindep = 1e-12[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m eigh level_shift = 0[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m eigh max_cycle = 100[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436380)[0m chkfile = molecule_alone.chk[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m Excited State energies (eV)
[36m(calculate_point_effect_cpu pid=3436377)[0m [7.46844092]
[36m(calculate_point_effect_cpu pid=3436385)[0m Running TDDFT in current process (force_single_gpu=False)[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m ******** <class 'pyscf.tdscf.rks.TDDFT'> for <class 'pyscf.qmmm.itrf.QMMMRKS'> ********[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m max_memory 4000 MB (current use 250 MB)[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m   HOMO = -0.287129688167752  LUMO = 0.0390879399167207[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m cycle= 7 E= -155.028451774925  delta_E= -1.75e-10  |g|= 2.85e-06  |ddm|= 3.49e-05[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m converged SCF energy = -155.028451774928
[36m(calculate_point_effect_cpu pid=3436385)[0m nstates = 1 singlet
[36m(calculate_point_effect_cpu pid=3436385)[0m deg_eia_thresh = 1.000e-03
[36m(calculate_point_effect_cpu pid=3436385)[0m wfnsym = None
[36m(calculate_point_effect_cpu pid=3436385)[0m conv_tol = 1e-05
[36m(calculate_point_effect_cpu pid=3436385)[0m eigh lindep = 1e-12
[36m(calculate_point_effect_cpu pid=3436385)[0m eigh level_shift = 0
[36m(calculate_point_effect_cpu pid=3436385)[0m eigh max_cycle = 100
[36m(calculate_point_effect_cpu pid=3436385)[0m chkfile = molecule_alone.chk
[36m(calculate_point_effect_cpu pid=3436375)[0m [7.4758426]
[36m(calculate_point_effect_cpu pid=3436381)[0m [7.46860097]
[36m(calculate_point_effect_cpu pid=3436379)[0m [6.98869483]
[36m(calculate_point_effect_cpu pid=3436383)[0m [7.1832533]
[36m(calculate_point_effect_cpu pid=3436389)[0m [7.05328598]
[36m(calculate_point_effect_cpu pid=3436378)[0m [7.12787148]
[36m(calculate_point_effect_cpu pid=3436378)[0m Excited State energies (eV)[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436387)[0m [7.41493151]
[36m(calculate_point_effect_cpu pid=3436387)[0m Excited State energies (eV)
[36m(calculate_point_effect_cpu pid=3436384)[0m Excited State energies (eV)
[36m(calculate_point_effect_cpu pid=3436384)[0m [7.46920091]
[36m(calculate_point_effect_cpu pid=3436388)[0m [7.30788435]
[36m(calculate_point_effect_cpu pid=3436388)[0m Excited State energies (eV)
[36m(calculate_point_effect_cpu pid=3436380)[0m Excited State energies (eV)
[36m(calculate_point_effect_cpu pid=3436380)[0m [7.40699058]
[36m(calculate_point_effect_cpu pid=3436385)[0m [7.32761439]
[36m(calculate_point_effect_cpu pid=3436390)[0m [7.33618393]
[36m(calculate_point_effect_cpu pid=3436376)[0m [7.52492708]
[36m(calculate_point_effect_cpu pid=3436382)[0m [7.57205709]
[36m(calculate_point_effect_cpu pid=3436386)[0m [7.50389413]
[36m(calculate_point_effect_cpu pid=3436386)[0m Excited State energies (eV)[32m [repeated 5x across cluster][0m
Point 33/67: SUCCESS (33/67 total)
Point 34/67: SUCCESS (34/67 total)
Point 35/67: SUCCESS (35/67 total)
Point 36/67: SUCCESS (36/67 total)
Point 37/67: SUCCESS (37/67 total)
Point 38/67: SUCCESS (38/67 total)
Point 39/67: SUCCESS (39/67 total)
Point 40/67: SUCCESS (40/67 total)
Point 41/67: SUCCESS (41/67 total)
Point 42/67: SUCCESS (42/67 total)
Point 43/67: SUCCESS (43/67 total)
Point 44/67: SUCCESS (44/67 total)
Point 45/67: SUCCESS (45/67 total)
Point 46/67: SUCCESS (46/67 total)
Point 47/67: SUCCESS (47/67 total)
Point 48/67: SUCCESS (48/67 total)
[36m(calculate_point_effect_cpu pid=3436376)[0m [Point 50] Running on CPU cores: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, PID: 3436376
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m === Resurrecting molecule_alone.chk ===
[36m(calculate_point_effect_cpu pid=3436376)[0m Loaded XC functional: b3lyp
[36m(calculate_point_effect_cpu pid=3436376)[0m Creating DFT object with xc=b3lyp
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m ******** <class 'pyscf.dft.rks.RKS'> ********
[36m(calculate_point_effect_cpu pid=3436375)[0m method = RKS
[36m(calculate_point_effect_cpu pid=3436375)[0m initial guess = chkfile
[36m(calculate_point_effect_cpu pid=3436375)[0m damping factor = 0
[36m(calculate_point_effect_cpu pid=3436375)[0m level_shift factor = 0
[36m(calculate_point_effect_cpu pid=3436375)[0m DIIS = <class 'pyscf.scf.diis.CDIIS'>
[36m(calculate_point_effect_cpu pid=3436375)[0m diis_start_cycle = 1
[36m(calculate_point_effect_cpu pid=3436375)[0m diis_space = 8
[36m(calculate_point_effect_cpu pid=3436375)[0m diis_damp = 0
[36m(calculate_point_effect_cpu pid=3436375)[0m SCF conv_tol = 1e-09
[36m(calculate_point_effect_cpu pid=3436375)[0m SCF conv_tol_grad = None
[36m(calculate_point_effect_cpu pid=3436375)[0m SCF max_cycles = 50
[36m(calculate_point_effect_cpu pid=3436375)[0m direct_scf = True
[36m(calculate_point_effect_cpu pid=3436375)[0m direct_scf_tol = 1e-13
[36m(calculate_point_effect_cpu pid=3436375)[0m chkfile to save SCF result = /tmp/tmpg4wkcdu1.chk
[36m(calculate_point_effect_cpu pid=3436375)[0m max_memory 4000 MB (current use 218 MB)
[36m(calculate_point_effect_cpu pid=3436375)[0m XC library pyscf.dft.libxc version 7.0.0
[36m(calculate_point_effect_cpu pid=3436375)[0m     S. Lehtola, C. Steigemann, M. J.T. Oliveira, and M. A.L. Marques.,  SoftwareX 7, 1–5 (2018)
[36m(calculate_point_effect_cpu pid=3436375)[0m XC functionals = b3lyp
[36m(calculate_point_effect_cpu pid=3436375)[0m     P. J. Stephens, F. J. Devlin, C. F. Chabalowski, and M. J. Frisch.,  J. Phys. Chem. 98, 11623 (1994)
[36m(calculate_point_effect_cpu pid=3436375)[0m small_rho_cutoff = 1e-07
[36m(calculate_point_effect_cpu pid=3436375)[0m Set gradient conv threshold to 3.16228e-05
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m chkfile to save SCF result = /tmp/tmpjuulnwkx.chk
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m init E= -155.028768605456
[36m(calculate_point_effect_cpu pid=3436381)[0m   HOMO = -0.260574538961987  LUMO = 0.0672407765319121
[36m(calculate_point_effect_cpu pid=3436381)[0m cycle= 1 E= -155.028768605452  delta_E= 3.92e-12  |g|= 6.17e-06  |ddm|= 8.04e-06
[36m(calculate_point_effect_cpu pid=3436381)[0m Extra cycle  E= -155.028768605444  delta_E= 8.7e-12  |g|= 9.15e-06  |ddm|= 1.07e-05
[36m(calculate_point_effect_cpu pid=3436381)[0m converged SCF energy = -155.028768605444
[36m(calculate_point_effect_cpu pid=3436381)[0m Resurrected CPU DFT object: <class 'pyscf.dft.rks.RKS'>, Energy: -155.02876860544364
[36m(calculate_point_effect_cpu pid=3436381)[0m Running TDDFT in current process (force_single_gpu=b3lyp)
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m ******** <class 'pyscf.tdscf.rks.TDDFT'> for <class 'pyscf.dft.rks.RKS'> ********
[36m(calculate_point_effect_cpu pid=3436381)[0m nstates = 1 singlet
[36m(calculate_point_effect_cpu pid=3436381)[0m deg_eia_thresh = 1.000e-03
[36m(calculate_point_effect_cpu pid=3436381)[0m wfnsym = None
[36m(calculate_point_effect_cpu pid=3436381)[0m conv_tol = 1e-05
[36m(calculate_point_effect_cpu pid=3436381)[0m eigh lindep = 1e-12
[36m(calculate_point_effect_cpu pid=3436381)[0m eigh level_shift = 0
[36m(calculate_point_effect_cpu pid=3436381)[0m eigh max_cycle = 100
[36m(calculate_point_effect_cpu pid=3436381)[0m chkfile = molecule_alone.chk
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m ******** <class 'pyscf.qmmm.itrf.QMMMRKS'> ********
[36m(calculate_point_effect_cpu pid=3436381)[0m method = QMMMRKS
[36m(calculate_point_effect_cpu pid=3436381)[0m chkfile to save SCF result = molecule_alone.chk
[36m(calculate_point_effect_cpu pid=3436381)[0m ** Add background charges for QMMMRKS **
[36m(calculate_point_effect_cpu pid=3436381)[0m Excited State energies (eV)
[36m(calculate_point_effect_cpu pid=3436385)[0m [Point 52] Running on CPU cores: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, PID: 3436385[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m === Resurrecting molecule_alone.chk ===[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m Loaded XC functional: b3lyp[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m Creating DFT object with xc=b3lyp[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436376)[0m ******** <class 'pyscf.dft.rks.RKS'> ********[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436376)[0m method = RKS[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m initial guess = chkfile[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m damping factor = 0[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m level_shift factor = 0[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m DIIS = <class 'pyscf.scf.diis.CDIIS'>[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m diis_start_cycle = 1[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m diis_space = 8[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m diis_damp = 0[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m SCF conv_tol = 1e-09[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m SCF conv_tol_grad = None[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m SCF max_cycles = 50[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m direct_scf = True[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m direct_scf_tol = 1e-13[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436376)[0m chkfile to save SCF result = /tmp/tmpq_k8z13m.chk[32m [repeated 14x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m max_memory 4000 MB (current use 220 MB)[32m [repeated 32x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m XC library pyscf.dft.libxc version 7.0.0[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m     S. Lehtola, C. Steigemann, M. J.T. Oliveira, and M. A.L. Marques.,  SoftwareX 7, 1–5 (2018)[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m XC functionals = b3lyp[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m     P. J. Stephens, F. J. Devlin, C. F. Chabalowski, and M. J. Frisch.,  J. Phys. Chem. 98, 11623 (1994)[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m small_rho_cutoff = 1e-07[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436381)[0m Set gradient conv threshold to 3.16228e-05[32m [repeated 16x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m init E= -155.028768605456[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m   HOMO = -0.260572890409526  LUMO = 0.0672412713481207[32m [repeated 31x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m cycle= 1 E= -155.028768605452  delta_E= 3.92e-12  |g|= 6.17e-06  |ddm|= 8.04e-06[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m Extra cycle  E= -155.028768605444  delta_E= 8.7e-12  |g|= 9.15e-06  |ddm|= 1.07e-05[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m converged SCF energy = -155.028768605444[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m Resurrected CPU DFT object: <class 'pyscf.dft.rks.RKS'>, Energy: -155.02876860544364[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m Running TDDFT in current process (force_single_gpu=b3lyp)[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m ******** <class 'pyscf.tdscf.rks.TDDFT'> for <class 'pyscf.dft.rks.RKS'> ********[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m nstates = 1 singlet[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m deg_eia_thresh = 1.000e-03[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m wfnsym = None[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m conv_tol = 1e-05[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m eigh lindep = 1e-12[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m eigh level_shift = 0[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m eigh max_cycle = 100[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m chkfile = molecule_alone.chk[32m [repeated 15x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436389)[0m Excited State energies (eV)
[36m(calculate_point_effect_cpu pid=3436389)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m Running TDDFT in current process (force_single_gpu=False)
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m ******** <class 'pyscf.tdscf.rks.TDDFT'> for <class 'pyscf.qmmm.itrf.QMMMRKS'> ********
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436381)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436379)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m ******** <class 'pyscf.qmmm.itrf.QMMMRKS'> ********[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m method = QMMMRKS[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m chkfile to save SCF result = molecule_alone.chk[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m ** Add background charges for QMMMRKS **[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m initial guess = chkfile[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m damping factor = 0[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m level_shift factor = 0[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m DIIS = <class 'pyscf.scf.diis.CDIIS'>[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m diis_start_cycle = 1[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m diis_space = 8[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m diis_damp = 0[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m SCF conv_tol = 1e-09[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m SCF conv_tol_grad = None[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m SCF max_cycles = 50[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m direct_scf = True[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m direct_scf_tol = 1e-13[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436389)[0m max_memory 4000 MB (current use 251 MB)[32m [repeated 10x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m XC library pyscf.dft.libxc version 7.0.0[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m     S. Lehtola, C. Steigemann, M. J.T. Oliveira, and M. A.L. Marques.,  SoftwareX 7, 1–5 (2018)[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m XC functionals = b3lyp[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m     P. J. Stephens, F. J. Devlin, C. F. Chabalowski, and M. J. Frisch.,  J. Phys. Chem. 98, 11623 (1994)[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m small_rho_cutoff = 1e-07[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m Set gradient conv threshold to 3.16228e-05[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m init E= -155.026868074562[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m   HOMO = -0.283353017414727  LUMO = 0.0428514928521428[32m [repeated 56x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m cycle= 4 E= -155.027329700994  delta_E= -1.69e-06  |g|= 0.000439  |ddm|= 0.00249[32m [repeated 48x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436389)[0m Extra cycle  E= -155.027862752942  delta_E= -1.14e-12  |g|= 1.44e-06  |ddm|= 4.35e-06[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436389)[0m converged SCF energy = -155.027862752942[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436389)[0m nstates = 1 singlet[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436389)[0m deg_eia_thresh = 1.000e-03[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436389)[0m wfnsym = None[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436389)[0m conv_tol = 1e-05[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436389)[0m eigh lindep = 1e-12[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436389)[0m eigh level_shift = 0[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436389)[0m eigh max_cycle = 100[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436389)[0m chkfile = molecule_alone.chk[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436383)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436375)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436387)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436378)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m Excited State energies (eV)[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436377)[0m Running TDDFT in current process (force_single_gpu=False)[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m ******** <class 'pyscf.tdscf.rks.TDDFT'> for <class 'pyscf.qmmm.itrf.QMMMRKS'> ********[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m ******** <class 'pyscf.qmmm.itrf.QMMMRKS'> ********
[36m(calculate_point_effect_cpu pid=3436385)[0m method = QMMMRKS
[36m(calculate_point_effect_cpu pid=3436385)[0m chkfile to save SCF result = molecule_alone.chk
[36m(calculate_point_effect_cpu pid=3436385)[0m ** Add background charges for QMMMRKS **
[36m(calculate_point_effect_cpu pid=3436385)[0m initial guess = chkfile
[36m(calculate_point_effect_cpu pid=3436385)[0m damping factor = 0
[36m(calculate_point_effect_cpu pid=3436385)[0m level_shift factor = 0
[36m(calculate_point_effect_cpu pid=3436385)[0m DIIS = <class 'pyscf.scf.diis.CDIIS'>
[36m(calculate_point_effect_cpu pid=3436385)[0m diis_start_cycle = 1
[36m(calculate_point_effect_cpu pid=3436385)[0m diis_space = 8
[36m(calculate_point_effect_cpu pid=3436385)[0m diis_damp = 0
[36m(calculate_point_effect_cpu pid=3436385)[0m SCF conv_tol = 1e-09
[36m(calculate_point_effect_cpu pid=3436385)[0m SCF conv_tol_grad = None
[36m(calculate_point_effect_cpu pid=3436385)[0m SCF max_cycles = 50
[36m(calculate_point_effect_cpu pid=3436385)[0m direct_scf = True
[36m(calculate_point_effect_cpu pid=3436385)[0m direct_scf_tol = 1e-13
[36m(calculate_point_effect_cpu pid=3436385)[0m max_memory 4000 MB (current use 217 MB)[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m XC library pyscf.dft.libxc version 7.0.0
[36m(calculate_point_effect_cpu pid=3436385)[0m     S. Lehtola, C. Steigemann, M. J.T. Oliveira, and M. A.L. Marques.,  SoftwareX 7, 1–5 (2018)
[36m(calculate_point_effect_cpu pid=3436385)[0m XC functionals = b3lyp
[36m(calculate_point_effect_cpu pid=3436385)[0m     P. J. Stephens, F. J. Devlin, C. F. Chabalowski, and M. J. Frisch.,  J. Phys. Chem. 98, 11623 (1994)
[36m(calculate_point_effect_cpu pid=3436385)[0m small_rho_cutoff = 1e-07
[36m(calculate_point_effect_cpu pid=3436385)[0m Set gradient conv threshold to 3.16228e-05
[36m(calculate_point_effect_cpu pid=3436377)[0m   HOMO = -0.283350841603425  LUMO = 0.0428540167625175[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m cycle= 7 E= -155.027329740566  delta_E= -1.01e-10  |g|= 2.44e-06  |ddm|= 2.35e-05[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m Extra cycle  E= -155.027329740567  delta_E= -7.39e-13  |g|= 1.38e-06  |ddm|= 4.98e-06[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m converged SCF energy = -155.027329740567[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m nstates = 1 singlet[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m deg_eia_thresh = 1.000e-03[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m wfnsym = None[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m conv_tol = 1e-05[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m eigh lindep = 1e-12[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m eigh level_shift = 0[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m eigh max_cycle = 100[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436377)[0m chkfile = molecule_alone.chk[32m [repeated 5x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m Excited State energies (eV)
[36m(calculate_point_effect_cpu pid=3436385)[0m init E= -155.027744225137
[36m(calculate_point_effect_cpu pid=3436388)[0m Excited State energies (eV)
[36m(calculate_point_effect_cpu pid=3436388)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m ******** <class 'pyscf.qmmm.itrf.QMMMRKS'> ********
[36m(calculate_point_effect_cpu pid=3436388)[0m method = QMMMRKS
[36m(calculate_point_effect_cpu pid=3436388)[0m initial guess = chkfile
[36m(calculate_point_effect_cpu pid=3436388)[0m damping factor = 0
[36m(calculate_point_effect_cpu pid=3436388)[0m level_shift factor = 0
[36m(calculate_point_effect_cpu pid=3436388)[0m DIIS = <class 'pyscf.scf.diis.CDIIS'>
[36m(calculate_point_effect_cpu pid=3436388)[0m diis_start_cycle = 1
[36m(calculate_point_effect_cpu pid=3436388)[0m diis_space = 8
[36m(calculate_point_effect_cpu pid=3436388)[0m diis_damp = 0
[36m(calculate_point_effect_cpu pid=3436388)[0m SCF conv_tol = 1e-09
[36m(calculate_point_effect_cpu pid=3436388)[0m SCF conv_tol_grad = None
[36m(calculate_point_effect_cpu pid=3436388)[0m SCF max_cycles = 50
[36m(calculate_point_effect_cpu pid=3436388)[0m direct_scf = True
[36m(calculate_point_effect_cpu pid=3436388)[0m direct_scf_tol = 1e-13
[36m(calculate_point_effect_cpu pid=3436388)[0m chkfile to save SCF result = molecule_alone.chk
[36m(calculate_point_effect_cpu pid=3436388)[0m XC library pyscf.dft.libxc version 7.0.0
[36m(calculate_point_effect_cpu pid=3436388)[0m     S. Lehtola, C. Steigemann, M. J.T. Oliveira, and M. A.L. Marques.,  SoftwareX 7, 1–5 (2018)
[36m(calculate_point_effect_cpu pid=3436388)[0m XC functionals = b3lyp
[36m(calculate_point_effect_cpu pid=3436388)[0m     P. J. Stephens, F. J. Devlin, C. F. Chabalowski, and M. J. Frisch.,  J. Phys. Chem. 98, 11623 (1994)
[36m(calculate_point_effect_cpu pid=3436388)[0m small_rho_cutoff = 1e-07
[36m(calculate_point_effect_cpu pid=3436388)[0m ** Add background charges for QMMMRKS **
[36m(calculate_point_effect_cpu pid=3436388)[0m Set gradient conv threshold to 3.16228e-05
[36m(calculate_point_effect_cpu pid=3436380)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m max_memory 4000 MB (current use 277 MB)[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436390)[0m   HOMO = -0.276909488648134  LUMO = 0.0513585861644116[32m [repeated 13x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436386)[0m cycle= 1 E= -155.022804982833  delta_E= -0.000336  |g|= 0.0181  |ddm|= 0.0937[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m init E= -155.021986202535[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m Extra cycle  E= -155.028195086648  delta_E= -2.56e-13  |g|= 1.41e-06  |ddm|= 3.89e-06
[36m(calculate_point_effect_cpu pid=3436385)[0m converged SCF energy = -155.028195086648
[36m(calculate_point_effect_cpu pid=3436385)[0m Running TDDFT in current process (force_single_gpu=False)
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m ******** <class 'pyscf.tdscf.rks.TDDFT'> for <class 'pyscf.qmmm.itrf.QMMMRKS'> ********
[36m(calculate_point_effect_cpu pid=3436385)[0m nstates = 1 singlet
[36m(calculate_point_effect_cpu pid=3436385)[0m deg_eia_thresh = 1.000e-03
[36m(calculate_point_effect_cpu pid=3436385)[0m wfnsym = None
[36m(calculate_point_effect_cpu pid=3436385)[0m conv_tol = 1e-05
[36m(calculate_point_effect_cpu pid=3436385)[0m eigh lindep = 1e-12
[36m(calculate_point_effect_cpu pid=3436385)[0m eigh level_shift = 0
[36m(calculate_point_effect_cpu pid=3436385)[0m eigh max_cycle = 100
[36m(calculate_point_effect_cpu pid=3436385)[0m chkfile = molecule_alone.chk
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m Excited State energies (eV)[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m ******** <class 'pyscf.qmmm.itrf.QMMMRKS'> ********[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m method = QMMMRKS[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m initial guess = chkfile[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m damping factor = 0[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m level_shift factor = 0[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m DIIS = <class 'pyscf.scf.diis.CDIIS'>[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m diis_start_cycle = 1[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m diis_space = 8[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m diis_damp = 0[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m SCF conv_tol = 1e-09[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m SCF conv_tol_grad = None[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m SCF max_cycles = 50[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m direct_scf = True[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m direct_scf_tol = 1e-13[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m chkfile to save SCF result = molecule_alone.chk[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m XC library pyscf.dft.libxc version 7.0.0[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m     S. Lehtola, C. Steigemann, M. J.T. Oliveira, and M. A.L. Marques.,  SoftwareX 7, 1–5 (2018)[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m XC functionals = b3lyp[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m     P. J. Stephens, F. J. Devlin, C. F. Chabalowski, and M. J. Frisch.,  J. Phys. Chem. 98, 11623 (1994)[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m small_rho_cutoff = 1e-07[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m ** Add background charges for QMMMRKS **[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m Set gradient conv threshold to 3.16228e-05[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436386)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436388)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436380)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436390)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m
[36m(calculate_point_effect_cpu pid=3436376)[0m max_memory 4000 MB (current use 276 MB)[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m   HOMO = -0.283458678373992  LUMO = 0.0310454222478984[32m [repeated 48x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m cycle= 6 E= -155.022374620032  delta_E= -4.28e-09  |g|= 1.88e-05  |ddm|= 0.000125[32m [repeated 47x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436389)[0m [7.45421603]
[36m(calculate_point_effect_cpu pid=3436382)[0m Extra cycle  E= -155.022374620124  delta_E= -4.55e-13  |g|= 2.21e-06  |ddm|= 5.35e-06[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m converged SCF energy = -155.022374620124[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m Running TDDFT in current process (force_single_gpu=False)[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m ******** <class 'pyscf.tdscf.rks.TDDFT'> for <class 'pyscf.qmmm.itrf.QMMMRKS'> ********[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m nstates = 1 singlet[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m deg_eia_thresh = 1.000e-03[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m wfnsym = None[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m conv_tol = 1e-05[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m eigh lindep = 1e-12[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m eigh level_shift = 0[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m eigh max_cycle = 100[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m chkfile = molecule_alone.chk[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436389)[0m Excited State energies (eV)
[36m(calculate_point_effect_cpu pid=3436382)[0m max_memory 4000 MB (current use 278 MB)
[36m(calculate_point_effect_cpu pid=3436382)[0m   HOMO = -0.283458873151255  LUMO = 0.0310457129625827
[36m(calculate_point_effect_cpu pid=3436382)[0m cycle= 7 E= -155.022374620124  delta_E= -9.13e-11  |g|= 2.59e-06  |ddm|= 2.18e-05
[36m(calculate_point_effect_cpu pid=3436383)[0m Excited State energies (eV)
[36m(calculate_point_effect_cpu pid=3436383)[0m [7.45216423]
[36m(calculate_point_effect_cpu pid=3436381)[0m [7.26413923]
[36m(calculate_point_effect_cpu pid=3436379)[0m [7.46636668]
[36m(calculate_point_effect_cpu pid=3436375)[0m [7.47532708]
[36m(calculate_point_effect_cpu pid=3436387)[0m [7.44603293]
[36m(calculate_point_effect_cpu pid=3436377)[0m [7.36726819]
[36m(calculate_point_effect_cpu pid=3436378)[0m [7.45948315]
[36m(calculate_point_effect_cpu pid=3436378)[0m Excited State energies (eV)[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436386)[0m [7.04816352]
[36m(calculate_point_effect_cpu pid=3436386)[0m Excited State energies (eV)
[36m(calculate_point_effect_cpu pid=3436380)[0m Excited State energies (eV)
[36m(calculate_point_effect_cpu pid=3436380)[0m [7.44912894]
[36m(calculate_point_effect_cpu pid=3436390)[0m [7.45464658]
[36m(calculate_point_effect_cpu pid=3436388)[0m [7.45306222]
[36m(calculate_point_effect_cpu pid=3436376)[0m [7.46048658]
[36m(calculate_point_effect_cpu pid=3436376)[0m Excited State energies (eV)[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m [7.45976351]
[36m(calculate_point_effect_cpu pid=3436384)[0m [7.44730553]
Point 49/67: SUCCESS (49/67 total)
[36m(calculate_point_effect_cpu pid=3436382)[0m [6.99237759]
[36m(calculate_point_effect_cpu pid=3436382)[0m Excited State energies (eV)[32m [repeated 3x across cluster][0m
Point 50/67: SUCCESS (50/67 total)
Point 51/67: SUCCESS (51/67 total)
Point 52/67: SUCCESS (52/67 total)
Point 53/67: SUCCESS (53/67 total)
Point 54/67: SUCCESS (54/67 total)
Point 55/67: SUCCESS (55/67 total)
Point 56/67: SUCCESS (56/67 total)
Point 57/67: SUCCESS (57/67 total)
Point 58/67: SUCCESS (58/67 total)
Point 59/67: SUCCESS (59/67 total)
Point 60/67: SUCCESS (60/67 total)
Point 61/67: SUCCESS (61/67 total)
Point 62/67: SUCCESS (62/67 total)
Point 63/67: SUCCESS (63/67 total)
Point 64/67: SUCCESS (64/67 total)
[36m(calculate_point_effect_cpu pid=3436382)[0m [Point 64] Running on CPU cores: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, PID: 3436382
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m === Resurrecting molecule_alone.chk ===
[36m(calculate_point_effect_cpu pid=3436382)[0m Loaded XC functional: b3lyp
[36m(calculate_point_effect_cpu pid=3436382)[0m Creating DFT object with xc=b3lyp
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m ******** <class 'pyscf.dft.rks.RKS'> ********
[36m(calculate_point_effect_cpu pid=3436382)[0m method = RKS
[36m(calculate_point_effect_cpu pid=3436382)[0m initial guess = chkfile
[36m(calculate_point_effect_cpu pid=3436382)[0m damping factor = 0
[36m(calculate_point_effect_cpu pid=3436382)[0m level_shift factor = 0
[36m(calculate_point_effect_cpu pid=3436382)[0m DIIS = <class 'pyscf.scf.diis.CDIIS'>
[36m(calculate_point_effect_cpu pid=3436382)[0m diis_start_cycle = 1
[36m(calculate_point_effect_cpu pid=3436382)[0m diis_space = 8
[36m(calculate_point_effect_cpu pid=3436382)[0m diis_damp = 0
[36m(calculate_point_effect_cpu pid=3436382)[0m SCF conv_tol = 1e-09
[36m(calculate_point_effect_cpu pid=3436382)[0m SCF conv_tol_grad = None
[36m(calculate_point_effect_cpu pid=3436382)[0m SCF max_cycles = 50
[36m(calculate_point_effect_cpu pid=3436382)[0m direct_scf = True
[36m(calculate_point_effect_cpu pid=3436382)[0m direct_scf_tol = 1e-13
[36m(calculate_point_effect_cpu pid=3436382)[0m chkfile to save SCF result = /tmp/tmpsf5dda67.chk
[36m(calculate_point_effect_cpu pid=3436382)[0m max_memory 4000 MB (current use 278 MB)
[36m(calculate_point_effect_cpu pid=3436382)[0m XC library pyscf.dft.libxc version 7.0.0
[36m(calculate_point_effect_cpu pid=3436382)[0m     S. Lehtola, C. Steigemann, M. J.T. Oliveira, and M. A.L. Marques.,  SoftwareX 7, 1–5 (2018)
[36m(calculate_point_effect_cpu pid=3436382)[0m XC functionals = b3lyp
[36m(calculate_point_effect_cpu pid=3436382)[0m     P. J. Stephens, F. J. Devlin, C. F. Chabalowski, and M. J. Frisch.,  J. Phys. Chem. 98, 11623 (1994)
[36m(calculate_point_effect_cpu pid=3436382)[0m small_rho_cutoff = 1e-07
[36m(calculate_point_effect_cpu pid=3436382)[0m Set gradient conv threshold to 3.16228e-05
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m init E= -155.028768605456
[36m(calculate_point_effect_cpu pid=3436382)[0m   HOMO = -0.260574538961987  LUMO = 0.0672407765319121
[36m(calculate_point_effect_cpu pid=3436382)[0m cycle= 1 E= -155.028768605452  delta_E= 3.92e-12  |g|= 6.17e-06  |ddm|= 8.04e-06
[36m(calculate_point_effect_cpu pid=3436382)[0m   HOMO = -0.260572890409526  LUMO = 0.0672412713481207
[36m(calculate_point_effect_cpu pid=3436382)[0m Extra cycle  E= -155.028768605444  delta_E= 8.7e-12  |g|= 9.15e-06  |ddm|= 1.07e-05
[36m(calculate_point_effect_cpu pid=3436382)[0m converged SCF energy = -155.028768605444
[36m(calculate_point_effect_cpu pid=3436382)[0m Resurrected CPU DFT object: <class 'pyscf.dft.rks.RKS'>, Energy: -155.02876860544364
[36m(calculate_point_effect_cpu pid=3436382)[0m Running TDDFT in current process (force_single_gpu=b3lyp)
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m ******** <class 'pyscf.tdscf.rks.TDDFT'> for <class 'pyscf.dft.rks.RKS'> ********
[36m(calculate_point_effect_cpu pid=3436382)[0m nstates = 1 singlet
[36m(calculate_point_effect_cpu pid=3436382)[0m deg_eia_thresh = 1.000e-03
[36m(calculate_point_effect_cpu pid=3436382)[0m wfnsym = None
[36m(calculate_point_effect_cpu pid=3436382)[0m conv_tol = 1e-05
[36m(calculate_point_effect_cpu pid=3436382)[0m eigh lindep = 1e-12
[36m(calculate_point_effect_cpu pid=3436382)[0m eigh level_shift = 0
[36m(calculate_point_effect_cpu pid=3436382)[0m eigh max_cycle = 100
[36m(calculate_point_effect_cpu pid=3436382)[0m chkfile = molecule_alone.chk
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m ******** <class 'pyscf.qmmm.itrf.QMMMRKS'> ********
[36m(calculate_point_effect_cpu pid=3436382)[0m method = QMMMRKS
[36m(calculate_point_effect_cpu pid=3436382)[0m chkfile to save SCF result = molecule_alone.chk
[36m(calculate_point_effect_cpu pid=3436382)[0m ** Add background charges for QMMMRKS **
[36m(calculate_point_effect_cpu pid=3436382)[0m Excited State energies (eV)
[36m(calculate_point_effect_cpu pid=3436385)[0m [Point 66] Running on CPU cores: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, PID: 3436385[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m === Resurrecting molecule_alone.chk ===[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m Loaded XC functional: b3lyp[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m Creating DFT object with xc=b3lyp[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m ******** <class 'pyscf.dft.rks.RKS'> ********[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m method = RKS[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m initial guess = chkfile[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m damping factor = 0[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m level_shift factor = 0[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m DIIS = <class 'pyscf.scf.diis.CDIIS'>[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m diis_start_cycle = 1[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m diis_space = 8[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m diis_damp = 0[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m SCF conv_tol = 1e-09[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m SCF conv_tol_grad = None[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m SCF max_cycles = 50[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m direct_scf = True[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m direct_scf_tol = 1e-13[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m chkfile to save SCF result = /tmp/tmps_ipawx6.chk[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m max_memory 4000 MB (current use 275 MB)[32m [repeated 6x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m XC library pyscf.dft.libxc version 7.0.0[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m     S. Lehtola, C. Steigemann, M. J.T. Oliveira, and M. A.L. Marques.,  SoftwareX 7, 1–5 (2018)[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m XC functionals = b3lyp[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m     P. J. Stephens, F. J. Devlin, C. F. Chabalowski, and M. J. Frisch.,  J. Phys. Chem. 98, 11623 (1994)[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m small_rho_cutoff = 1e-07[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m Set gradient conv threshold to 3.16228e-05[32m [repeated 3x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m init E= -155.028768605456[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m   HOMO = -0.260572890409526  LUMO = 0.0672412713481207[32m [repeated 4x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m cycle= 1 E= -155.028768605452  delta_E= 3.92e-12  |g|= 6.17e-06  |ddm|= 8.04e-06[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m Extra cycle  E= -155.028768605444  delta_E= 8.7e-12  |g|= 9.15e-06  |ddm|= 1.07e-05[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m converged SCF energy = -155.028768605444[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m Resurrected CPU DFT object: <class 'pyscf.dft.rks.RKS'>, Energy: -155.02876860544364[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m Running TDDFT in current process (force_single_gpu=b3lyp)[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m ******** <class 'pyscf.tdscf.rks.TDDFT'> for <class 'pyscf.dft.rks.RKS'> ********[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m nstates = 1 singlet[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m deg_eia_thresh = 1.000e-03[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m wfnsym = None[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m conv_tol = 1e-05[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m eigh lindep = 1e-12[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m eigh level_shift = 0[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m eigh max_cycle = 100[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m chkfile = molecule_alone.chk[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m Running TDDFT in current process (force_single_gpu=False)
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m ******** <class 'pyscf.tdscf.rks.TDDFT'> for <class 'pyscf.qmmm.itrf.QMMMRKS'> ********
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436382)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m Excited State energies (eV)
[36m(calculate_point_effect_cpu pid=3436385)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436382)[0m max_memory 4000 MB (current use 275 MB)
[36m(calculate_point_effect_cpu pid=3436382)[0m init E= -155.027128609361
[36m(calculate_point_effect_cpu pid=3436382)[0m   HOMO = -0.281271419400132  LUMO = 0.0471399864317167[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m cycle= 7 E= -155.027624392292  delta_E= -6.6e-11  |g|= 2.03e-06  |ddm|= 2.03e-05[32m [repeated 7x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m Extra cycle  E= -155.027624392294  delta_E= -1.28e-12  |g|= 1.2e-06  |ddm|= 4.1e-06
[36m(calculate_point_effect_cpu pid=3436382)[0m converged SCF energy = -155.027624392294
[36m(calculate_point_effect_cpu pid=3436382)[0m nstates = 1 singlet
[36m(calculate_point_effect_cpu pid=3436382)[0m deg_eia_thresh = 1.000e-03
[36m(calculate_point_effect_cpu pid=3436382)[0m wfnsym = None
[36m(calculate_point_effect_cpu pid=3436382)[0m conv_tol = 1e-05
[36m(calculate_point_effect_cpu pid=3436382)[0m eigh lindep = 1e-12
[36m(calculate_point_effect_cpu pid=3436382)[0m eigh level_shift = 0
[36m(calculate_point_effect_cpu pid=3436382)[0m eigh max_cycle = 100
[36m(calculate_point_effect_cpu pid=3436382)[0m chkfile = molecule_alone.chk
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m ******** <class 'pyscf.qmmm.itrf.QMMMRKS'> ********
[36m(calculate_point_effect_cpu pid=3436385)[0m method = QMMMRKS
[36m(calculate_point_effect_cpu pid=3436385)[0m initial guess = chkfile
[36m(calculate_point_effect_cpu pid=3436385)[0m damping factor = 0
[36m(calculate_point_effect_cpu pid=3436385)[0m level_shift factor = 0
[36m(calculate_point_effect_cpu pid=3436385)[0m DIIS = <class 'pyscf.scf.diis.CDIIS'>
[36m(calculate_point_effect_cpu pid=3436385)[0m diis_start_cycle = 1
[36m(calculate_point_effect_cpu pid=3436385)[0m diis_space = 8
[36m(calculate_point_effect_cpu pid=3436385)[0m diis_damp = 0
[36m(calculate_point_effect_cpu pid=3436385)[0m SCF conv_tol = 1e-09
[36m(calculate_point_effect_cpu pid=3436385)[0m SCF conv_tol_grad = None
[36m(calculate_point_effect_cpu pid=3436385)[0m SCF max_cycles = 50
[36m(calculate_point_effect_cpu pid=3436385)[0m direct_scf = True
[36m(calculate_point_effect_cpu pid=3436385)[0m direct_scf_tol = 1e-13
[36m(calculate_point_effect_cpu pid=3436385)[0m chkfile to save SCF result = molecule_alone.chk
[36m(calculate_point_effect_cpu pid=3436385)[0m max_memory 4000 MB (current use 216 MB)
[36m(calculate_point_effect_cpu pid=3436385)[0m XC library pyscf.dft.libxc version 7.0.0
[36m(calculate_point_effect_cpu pid=3436385)[0m     S. Lehtola, C. Steigemann, M. J.T. Oliveira, and M. A.L. Marques.,  SoftwareX 7, 1–5 (2018)
[36m(calculate_point_effect_cpu pid=3436385)[0m XC functionals = b3lyp
[36m(calculate_point_effect_cpu pid=3436385)[0m     P. J. Stephens, F. J. Devlin, C. F. Chabalowski, and M. J. Frisch.,  J. Phys. Chem. 98, 11623 (1994)
[36m(calculate_point_effect_cpu pid=3436385)[0m small_rho_cutoff = 1e-07
[36m(calculate_point_effect_cpu pid=3436385)[0m ** Add background charges for QMMMRKS **
[36m(calculate_point_effect_cpu pid=3436385)[0m Set gradient conv threshold to 3.16228e-05
[36m(calculate_point_effect_cpu pid=3436384)[0m [7.40200813]
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m init E= -155.026169146013
[36m(calculate_point_effect_cpu pid=3436385)[0m Extra cycle  E= -155.026703532269  delta_E= -9.66e-13  |g|= 1.36e-06  |ddm|= 5.17e-06
[36m(calculate_point_effect_cpu pid=3436385)[0m converged SCF energy = -155.026703532269
[36m(calculate_point_effect_cpu pid=3436385)[0m Running TDDFT in current process (force_single_gpu=False)
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m ******** <class 'pyscf.tdscf.rks.TDDFT'> for <class 'pyscf.qmmm.itrf.QMMMRKS'> ********
[36m(calculate_point_effect_cpu pid=3436385)[0m nstates = 1 singlet
[36m(calculate_point_effect_cpu pid=3436385)[0m deg_eia_thresh = 1.000e-03
[36m(calculate_point_effect_cpu pid=3436385)[0m wfnsym = None
[36m(calculate_point_effect_cpu pid=3436385)[0m conv_tol = 1e-05
[36m(calculate_point_effect_cpu pid=3436385)[0m eigh lindep = 1e-12
[36m(calculate_point_effect_cpu pid=3436385)[0m eigh level_shift = 0
[36m(calculate_point_effect_cpu pid=3436385)[0m eigh max_cycle = 100
[36m(calculate_point_effect_cpu pid=3436385)[0m chkfile = molecule_alone.chk
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436385)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m Excited State energies (eV)
[36m(calculate_point_effect_cpu pid=3436384)[0m   HOMO = -0.284270397909766  LUMO = 0.0462115875737363[32m [repeated 10x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m cycle= 1 E= -155.027145574013  delta_E= -0.000397  |g|= 0.0173  |ddm|= 0.126[32m [repeated 8x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m ******** <class 'pyscf.qmmm.itrf.QMMMRKS'> ********
[36m(calculate_point_effect_cpu pid=3436384)[0m method = QMMMRKS
[36m(calculate_point_effect_cpu pid=3436384)[0m initial guess = chkfile
[36m(calculate_point_effect_cpu pid=3436384)[0m damping factor = 0
[36m(calculate_point_effect_cpu pid=3436384)[0m level_shift factor = 0
[36m(calculate_point_effect_cpu pid=3436384)[0m DIIS = <class 'pyscf.scf.diis.CDIIS'>
[36m(calculate_point_effect_cpu pid=3436384)[0m diis_start_cycle = 1
[36m(calculate_point_effect_cpu pid=3436384)[0m diis_space = 8
[36m(calculate_point_effect_cpu pid=3436384)[0m diis_damp = 0
[36m(calculate_point_effect_cpu pid=3436384)[0m SCF conv_tol = 1e-09
[36m(calculate_point_effect_cpu pid=3436384)[0m SCF conv_tol_grad = None
[36m(calculate_point_effect_cpu pid=3436384)[0m SCF max_cycles = 50
[36m(calculate_point_effect_cpu pid=3436384)[0m direct_scf = True
[36m(calculate_point_effect_cpu pid=3436384)[0m direct_scf_tol = 1e-13
[36m(calculate_point_effect_cpu pid=3436384)[0m chkfile to save SCF result = molecule_alone.chk
[36m(calculate_point_effect_cpu pid=3436385)[0m max_memory 4000 MB (current use 248 MB)[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m XC library pyscf.dft.libxc version 7.0.0
[36m(calculate_point_effect_cpu pid=3436384)[0m     S. Lehtola, C. Steigemann, M. J.T. Oliveira, and M. A.L. Marques.,  SoftwareX 7, 1–5 (2018)
[36m(calculate_point_effect_cpu pid=3436384)[0m XC functionals = b3lyp
[36m(calculate_point_effect_cpu pid=3436384)[0m     P. J. Stephens, F. J. Devlin, C. F. Chabalowski, and M. J. Frisch.,  J. Phys. Chem. 98, 11623 (1994)
[36m(calculate_point_effect_cpu pid=3436384)[0m small_rho_cutoff = 1e-07
[36m(calculate_point_effect_cpu pid=3436384)[0m ** Add background charges for QMMMRKS **
[36m(calculate_point_effect_cpu pid=3436384)[0m Set gradient conv threshold to 3.16228e-05
[36m(calculate_point_effect_cpu pid=3436384)[0m init E= -155.02674873321
[36m(calculate_point_effect_cpu pid=3436384)[0m   HOMO = -0.281921077904461  LUMO = 0.0464323818846855[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m cycle= 3 E= -155.027223816099  delta_E= -9.14e-05  |g|= 0.00361  |ddm|= 0.0198[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436382)[0m Excited State energies (eV)
[36m(calculate_point_effect_cpu pid=3436382)[0m [7.43932702]
[36m(calculate_point_effect_cpu pid=3436384)[0m   HOMO = -0.281891037125026  LUMO = 0.0464495982604772[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m cycle= 5 E= -155.027226491755  delta_E= -1.46e-08  |g|= 0.000122  |ddm|= 0.000359[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m Extra cycle  E= -155.027226495468  delta_E= -8.53e-13  |g|= 1.5e-06  |ddm|= 4.22e-06
[36m(calculate_point_effect_cpu pid=3436384)[0m converged SCF energy = -155.027226495468
[36m(calculate_point_effect_cpu pid=3436384)[0m Running TDDFT in current process (force_single_gpu=False)
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m ******** <class 'pyscf.tdscf.rks.TDDFT'> for <class 'pyscf.qmmm.itrf.QMMMRKS'> ********
[36m(calculate_point_effect_cpu pid=3436384)[0m nstates = 1 singlet
[36m(calculate_point_effect_cpu pid=3436384)[0m deg_eia_thresh = 1.000e-03
[36m(calculate_point_effect_cpu pid=3436384)[0m wfnsym = None
[36m(calculate_point_effect_cpu pid=3436384)[0m conv_tol = 1e-05
[36m(calculate_point_effect_cpu pid=3436384)[0m eigh lindep = 1e-12
[36m(calculate_point_effect_cpu pid=3436384)[0m eigh level_shift = 0
[36m(calculate_point_effect_cpu pid=3436384)[0m eigh max_cycle = 100
[36m(calculate_point_effect_cpu pid=3436384)[0m chkfile = molecule_alone.chk
[36m(calculate_point_effect_cpu pid=3436384)[0m max_memory 4000 MB (current use 276 MB)
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m
[36m(calculate_point_effect_cpu pid=3436384)[0m   HOMO = -0.281891545145645  LUMO = 0.0464487568939761[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436384)[0m cycle= 7 E= -155.027226495467  delta_E= -1.11e-10  |g|= 2.18e-06  |ddm|= 2.02e-05[32m [repeated 2x across cluster][0m
[36m(calculate_point_effect_cpu pid=3436385)[0m Excited State energies (eV)
[36m(calculate_point_effect_cpu pid=3436385)[0m [7.35250135]
[36m(calculate_point_effect_cpu pid=3436384)[0m [7.44067271]
[36m(calculate_point_effect_cpu pid=3436384)[0m Excited State energies (eV)
Point 65/67: SUCCESS (65/67 total)
Point 66/67: SUCCESS (66/67 total)
Point 67/67: SUCCESS (67/67 total)

Final statistics written to: logs_20260325_015003/calculation_summary.out
Raw properties appended to: logs_20260325_015003/calculation_summary.out
Created: CCO_opt2_s1_exe.mol2
Created: CCO_opt2_s1_exe_normalized.mol2

Created: CCO_opt2_tuning_summary.csv

Organizing results into: results_CCO_opt2_2026-03-25_02-01-42/
  Moved: CCO_opt2_tuning_summary.csv
  Moved 2 MOL2 files
  Added normalization parameters to summary
  Moved: logs_20260325_015003/ -> logs/

Fetching inspirational quote...


  The principle is sound. To avoid illness, expose yourself to germs, enabling your immune system to develop antibodies. I don’t know why everyone doesn’t do this… maybe they have something against living forever.
                     - Dwight Schrute
