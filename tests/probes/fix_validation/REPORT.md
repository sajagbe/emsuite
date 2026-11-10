# Fix validation report

Code under test: HEAD `5dbbe5c72ac62112d667fc694bd4432ba765d913` + `code_under_test.diff`

**Summary: 40 PASS, 2 FAIL, 7 MISSING**

| case | check | result |
|---|---|---|
| water_full | gpu vs reference | PASS |
| water_full | cpu vs reference | PASS |
| water_full | cpu vs gpu | PASS |
| water_full | gpu IE/EA baseline physical | PASS |
| water_full | cpu IE/EA baseline physical | PASS |
| lf_gas | gpu vs reference | PASS |
| lf_gas | cpu vs reference | PASS |
| lf_gas | cpu vs gpu | PASS |
| lf_gas | gpu IE/EA baseline physical | PASS |
| lf_gas | cpu IE/EA baseline physical | PASS |
| rPSB_gas | gpu vs reference | PASS |
| rPSB_gas | cpu vs reference | PASS |
| rPSB_gas | cpu vs gpu | PASS |
| rPSB_gas | gpu IE/EA baseline physical | PASS |
| rPSB_gas | cpu IE/EA baseline physical | PASS |
| SB_gas | gpu vs reference | PASS |
| SB_gas | cpu vs reference | FAIL |
| SB_gas | cpu vs gpu | FAIL |
| SB_gas | gpu IE/EA baseline physical | PASS |
| SB_gas | cpu IE/EA baseline physical | PASS |
| rPSB_nhexane | gpu vs reference | PASS |
| rPSB_nhexane | cpu vs reference | MISSING |
| rPSB_nhexane | cpu vs gpu | MISSING |
| rPSB_nhexane | gpu IE/EA baseline physical | PASS |
| rPSB_dmso | gpu vs reference | PASS |
| rPSB_dmso | cpu vs reference | MISSING |
| rPSB_dmso | cpu vs gpu | MISSING |
| rPSB_dmso | gpu IE/EA baseline physical | PASS |
| SB_water | gpu vs reference | MISSING |
| SB_water | cpu vs reference | MISSING |
| SB_water | cpu vs gpu | MISSING |
| baseline rPSB_gas | gpu | PASS |
| baseline rPSB_gas | cpu | PASS |
| baseline SB_gas | gpu | PASS |
| baseline SB_gas | cpu | PASS |
| baseline rPSB_nhexane | gpu | PASS |
| baseline rPSB_nhexane | cpu | PASS |
| baseline rPSB_chloroform | gpu | PASS |
| baseline rPSB_chloroform | cpu | PASS |
| baseline rPSB_acetonitrile | gpu | PASS |
| baseline rPSB_acetonitrile | cpu | PASS |
| baseline SB_dmso | gpu | PASS |
| baseline SB_dmso | cpu | PASS |
| baseline rPSB_gas | cpu vs gpu | PASS |
| baseline SB_gas | cpu vs gpu | PASS |
| baseline rPSB_nhexane | cpu vs gpu | PASS |
| baseline rPSB_chloroform | cpu vs gpu | PASS |
| baseline rPSB_acetonitrile | cpu vs gpu | PASS |
| baseline SB_dmso | cpu vs gpu | PASS |

## A. Tuning maps vs pre-drift references

### water_full

Reference: `/data/PHO_WORK/sajagbe2/packages/emsuite/examples/water-gpu/results_Water_2026-09-02_00-39-19/Water_tuning_summary.csv` points [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]  
Reference IE/EA baseline physical: **True**

**gpu vs reference: PASS** (24 columns)

| column | max diff | tolerance | status |
|---|---|---|---|
| cp_baseline | 0 | 0.05 abs | PASS |
| dm_baseline | 2.22e-15 | 0.001 abs | PASS |
| ea_baseline | 1.78e-11 | 0.05 abs | PASS |
| efl_baseline | 8.46e-14 | 0.002 abs | PASS |
| eng_baseline | 0 | 0.002 abs | PASS |
| gap_baseline | 4.86e-06 | 0.001 abs | PASS |
| gse_baseline | 0.0362 | 0.05 abs | PASS |
| hard_baseline | 7.73e-13 | 0.002 abs | PASS |
| homo_baseline | 4.03e-06 | 0.001 abs | PASS |
| ie_effect | 4.46e-11 | 0.05 abs | PASS |
| ie_baseline | 1.78e-11 | 0.05 abs | PASS |
| lumo_baseline | 8.26e-07 | 0.001 abs | PASS |
| nfl_baseline | 9.27e-14 | 0.001 rel | PASS |

**cpu vs reference: PASS** (24 columns)

| column | max diff | tolerance | status |
|---|---|---|---|
| cp_baseline | 9.36e-10 | 0.05 abs | PASS |
| dm_baseline | 5.57e-08 | 0.001 abs | PASS |
| ea_baseline | 9.1e-08 | 0.05 abs | PASS |
| efl_baseline | 4.46e-10 | 0.002 abs | PASS |
| eng_baseline | 4.06e-11 | 0.002 abs | PASS |
| gap_baseline | 1.25e-05 | 0.001 abs | PASS |
| gse_baseline | 0.0362 | 0.05 abs | PASS |
| hard_baseline | 3.91e-09 | 0.002 abs | PASS |
| homo_baseline | 1.86e-05 | 0.001 abs | PASS |
| ie_effect | 4.29e-08 | 0.05 abs | PASS |
| ie_baseline | 8.92e-08 | 0.05 abs | PASS |
| lumo_baseline | 6.13e-06 | 0.001 abs | PASS |
| nfl_baseline | 4.89e-10 | 0.001 rel | PASS |

**cpu vs gpu: PASS** (24 columns)

| column | max diff | tolerance | status |
|---|---|---|---|
| cp_baseline | 9.36e-10 | 0.05 abs | PASS |
| dm_baseline | 5.57e-08 | 0.001 abs | PASS |
| ea_baseline | 9.1e-08 | 0.05 abs | PASS |
| efl_baseline | 4.46e-10 | 0.002 abs | PASS |
| eng_baseline | 4.06e-11 | 0.002 abs | PASS |
| gap_baseline | 7.65e-06 | 0.001 abs | PASS |
| gse_baseline | 8.9e-08 | 0.05 abs | PASS |
| hard_baseline | 3.91e-09 | 0.002 abs | PASS |
| homo_baseline | 1.46e-05 | 0.001 abs | PASS |
| ie_effect | 4.3e-08 | 0.05 abs | PASS |
| ie_baseline | 8.91e-08 | 0.05 abs | PASS |
| lumo_baseline | 6.96e-06 | 0.001 abs | PASS |
| nfl_baseline | 4.88e-10 | 0.001 rel | PASS |

- new gpu baseline: IE 282.667, EA -102.544 kcal/mol (physical)
- new cpu baseline: IE 282.667, EA -102.544 kcal/mol (physical)

### lf_gas

Reference: `/data/PHO_WORK/sajagbe2/packages/emsuite/tests/integration_runs/recreate-maps-2026-10-07T035624Z/lf-gas/6-31Gs/separate/singlet/results_LF_2026-10-07_05-33-31/LF_tuning_summary.csv` points [7, 50, 52, 56, 71, 103, 154, 189]  
Reference IE/EA baseline physical: **True**

**gpu vs reference: PASS** (36 columns)

| column | max diff | tolerance | status |
|---|---|---|---|
| cp_baseline | 9.07e-08 | 0.05 abs | PASS |
| dm_baseline | 1.02e-11 | 0.001 abs | PASS |
| ea_baseline | 3.62e-08 | 0.05 abs | PASS |
| efl_baseline | 3.11e-09 | 0.002 abs | PASS |
| eng_baseline | 3.94e-09 | 0.002 abs | PASS |
| gap_baseline | 9.81e-14 | 0.001 abs | PASS |
| gse_baseline | 2.33e-10 | 0.05 abs | PASS |
| hard_baseline | 2.36e-09 | 0.002 abs | PASS |
| homo_baseline | 1.23e-13 | 0.001 abs | PASS |
| ie_effect | 1.48e-07 | 0.05 abs | PASS |
| ie_baseline | 1.45e-07 | 0.05 abs | PASS |
| lumo_baseline | 2.4e-14 | 0.001 abs | PASS |
| nfl_baseline | 1.09e-09 | 0.001 rel | PASS |
| s1_exe_effect | 5.08e-13 | 0.001 abs | PASS |
| s1_exe_baseline | 1.95e-05 | 0.001 abs | PASS |
| s1_osc_baseline | 3.37e-08 | 0.001 abs | PASS |
| s2_exe_effect | 1.88e-12 | 0.001 abs | PASS |
| s2_exe_baseline | 1.82e-05 | 0.001 abs | PASS |
| s2_osc_baseline | 5.03e-07 | 0.001 abs | PASS |
| s3_exe_effect | 3.02e-13 | 0.001 abs | PASS |
| s3_exe_baseline | 1.97e-05 | 0.001 abs | PASS |
| s3_osc_baseline | 2.63e-08 | 0.001 abs | PASS |

**cpu vs reference: PASS** (36 columns)

| column | max diff | tolerance | status |
|---|---|---|---|
| cp_baseline | 1.08e-07 | 0.05 abs | PASS |
| dm_baseline | 9.44e-05 | 0.001 abs | PASS |
| ea_baseline | 2.07e-07 | 0.05 abs | PASS |
| efl_baseline | 5.33e-09 | 0.002 abs | PASS |
| eng_baseline | 4.67e-09 | 0.002 abs | PASS |
| gap_baseline | 1.55e-05 | 0.001 abs | PASS |
| gse_baseline | 2.5e-07 | 0.05 abs | PASS |
| hard_baseline | 1.37e-08 | 0.002 abs | PASS |
| homo_baseline | 6.49e-06 | 0.001 abs | PASS |
| ie_effect | 5.84e-07 | 0.05 abs | PASS |
| ie_baseline | 4.23e-07 | 0.05 abs | PASS |
| lumo_baseline | 8.96e-06 | 0.001 abs | PASS |
| nfl_baseline | 1.87e-09 | 0.001 rel | PASS |
| s1_exe_effect | 3.38e-05 | 0.001 abs | PASS |
| s1_exe_baseline | 3.43e-07 | 0.001 abs | PASS |
| s1_osc_baseline | 4.17e-08 | 0.001 abs | PASS |
| s2_exe_effect | 8.56e-05 | 0.001 abs | PASS |
| s2_exe_baseline | 1.73e-06 | 0.001 abs | PASS |
| s2_osc_baseline | 2.66e-06 | 0.001 abs | PASS |
| s3_exe_effect | 0.000111 | 0.001 abs | PASS |
| s3_exe_baseline | 6.34e-07 | 0.001 abs | PASS |
| s3_osc_baseline | 1.47e-08 | 0.001 abs | PASS |

**cpu vs gpu: PASS** (36 columns)

| column | max diff | tolerance | status |
|---|---|---|---|
| cp_baseline | 1.68e-08 | 0.05 abs | PASS |
| dm_baseline | 9.44e-05 | 0.001 abs | PASS |
| ea_baseline | 2.44e-07 | 0.05 abs | PASS |
| efl_baseline | 8.44e-09 | 0.002 abs | PASS |
| eng_baseline | 7.3e-10 | 0.002 abs | PASS |
| gap_baseline | 1.55e-05 | 0.001 abs | PASS |
| gse_baseline | 2.5e-07 | 0.05 abs | PASS |
| hard_baseline | 1.13e-08 | 0.002 abs | PASS |
| homo_baseline | 6.49e-06 | 0.001 abs | PASS |
| ie_effect | 4.39e-07 | 0.05 abs | PASS |
| ie_baseline | 2.77e-07 | 0.05 abs | PASS |
| lumo_baseline | 8.96e-06 | 0.001 abs | PASS |
| nfl_baseline | 2.96e-09 | 0.001 rel | PASS |
| s1_exe_effect | 3.38e-05 | 0.001 abs | PASS |
| s1_exe_baseline | 1.98e-05 | 0.001 abs | PASS |
| s1_osc_baseline | 7.54e-08 | 0.001 abs | PASS |
| s2_exe_effect | 8.56e-05 | 0.001 abs | PASS |
| s2_exe_baseline | 1.64e-05 | 0.001 abs | PASS |
| s2_osc_baseline | 3.17e-06 | 0.001 abs | PASS |
| s3_exe_effect | 0.000111 | 0.001 abs | PASS |
| s3_exe_baseline | 1.91e-05 | 0.001 abs | PASS |
| s3_osc_baseline | 4.1e-08 | 0.001 abs | PASS |

- new gpu baseline: IE 181.117, EA 22.824 kcal/mol (physical)
- new cpu baseline: IE 181.117, EA 22.824 kcal/mol (physical)

### rPSB_gas

Reference: `/data/PHO_WORK/sajagbe2/packages/emsuite/tests/integration_runs/opsins/campaigns/pose-restrain-2026-10-11/gas/rPSB/separate_singlet/results_rPSB_2026-10-11_06-02-14/rPSB_tuning_summary.csv` points [57, 60, 126, 143, 163, 200]  
Reference IE/EA baseline physical: **True**

**gpu vs reference: PASS** (38 columns)

| column | max diff | tolerance | status |
|---|---|---|---|
| cp_baseline | 0 | 0.05 abs | PASS |
| dm_baseline | 2.68e-12 | 0.001 abs | PASS |
| ea_baseline | 5.71e-10 | 0.05 abs | PASS |
| efl_baseline | 9.7e-11 | 0.002 abs | PASS |
| eng_baseline | 0 | 0.002 abs | PASS |
| gap_baseline | 4.53e-14 | 0.001 abs | PASS |
| gse_baseline | 3.49e-10 | 0.05 abs | PASS |
| hard_baseline | 2.47e-11 | 0.002 abs | PASS |
| homo_baseline | 7.9e-14 | 0.001 abs | PASS |
| ie_effect | 1.71e-09 | 0.05 abs | PASS |
| ie_baseline | 5.71e-10 | 0.05 abs | PASS |
| lumo_baseline | 3.29e-14 | 0.001 abs | PASS |
| nfl_baseline | 9.92e-12 | 0.001 rel | PASS |
| s1_exe_effect | 9.84e-10 | 0.001 abs | PASS |
| s1_exe_baseline | 7.02e-10 | 0.001 abs | PASS |
| s1_osc_baseline | 1.23e-12 | 0.001 abs | PASS |
| s2_exe_effect | 9.54e-09 | 0.001 abs | PASS |
| s2_exe_baseline | 7.66e-09 | 0.001 abs | PASS |
| s2_osc_baseline | 7.04e-09 | 0.001 abs | PASS |
| s3_exe_effect | 2.12e-09 | 0.001 abs | PASS |
| s3_exe_baseline | 1.1e-09 | 0.001 abs | PASS |
| s3_osc_baseline | 9.11e-09 | 0.001 abs | PASS |
| spin_baseline | 0 | 0.001 abs | PASS |

**cpu vs reference: PASS** (38 columns)

| column | max diff | tolerance | status |
|---|---|---|---|
| cp_baseline | 2.81e-07 | 0.05 abs | PASS |
| dm_baseline | 0.000114 | 0.001 abs | PASS |
| ea_baseline | 1.31e-07 | 0.05 abs | PASS |
| efl_baseline | 8.68e-09 | 0.002 abs | PASS |
| eng_baseline | 1.22e-08 | 0.002 abs | PASS |
| gap_baseline | 0.000203 | 0.001 abs | PASS |
| gse_baseline | 3.73e-07 | 0.05 abs | PASS |
| hard_baseline | 6.48e-09 | 0.002 abs | PASS |
| homo_baseline | 0.000139 | 0.001 abs | PASS |
| ie_effect | 1.27e-06 | 0.05 abs | PASS |
| ie_baseline | 4.3e-07 | 0.05 abs | PASS |
| lumo_baseline | 6.32e-05 | 0.001 abs | PASS |
| nfl_baseline | 8.88e-10 | 0.001 rel | PASS |
| s1_exe_effect | 0.000177 | 0.001 abs | PASS |
| s1_exe_baseline | 0.000109 | 0.001 abs | PASS |
| s1_osc_baseline | 9.62e-08 | 0.001 abs | PASS |
| s2_exe_effect | 0.00017 | 0.001 abs | PASS |
| s2_exe_baseline | 7.52e-05 | 0.001 abs | PASS |
| s2_osc_baseline | 3.74e-05 | 0.001 abs | PASS |
| s3_exe_effect | 7.77e-05 | 0.001 abs | PASS |
| s3_exe_baseline | 7.43e-05 | 0.001 abs | PASS |
| s3_osc_baseline | 0.000228 | 0.001 abs | PASS |
| spin_baseline | 0 | 0.001 abs | PASS |

**cpu vs gpu: PASS** (38 columns)

| column | max diff | tolerance | status |
|---|---|---|---|
| cp_baseline | 2.81e-07 | 0.05 abs | PASS |
| dm_baseline | 0.000114 | 0.001 abs | PASS |
| ea_baseline | 1.32e-07 | 0.05 abs | PASS |
| efl_baseline | 8.78e-09 | 0.002 abs | PASS |
| eng_baseline | 1.22e-08 | 0.002 abs | PASS |
| gap_baseline | 0.000203 | 0.001 abs | PASS |
| gse_baseline | 3.73e-07 | 0.05 abs | PASS |
| hard_baseline | 6.46e-09 | 0.002 abs | PASS |
| homo_baseline | 0.000139 | 0.001 abs | PASS |
| ie_effect | 1.26e-06 | 0.05 abs | PASS |
| ie_baseline | 4.3e-07 | 0.05 abs | PASS |
| lumo_baseline | 6.32e-05 | 0.001 abs | PASS |
| nfl_baseline | 8.98e-10 | 0.001 rel | PASS |
| s1_exe_effect | 0.000177 | 0.001 abs | PASS |
| s1_exe_baseline | 0.000109 | 0.001 abs | PASS |
| s1_osc_baseline | 9.62e-08 | 0.001 abs | PASS |
| s2_exe_effect | 0.00017 | 0.001 abs | PASS |
| s2_exe_baseline | 7.52e-05 | 0.001 abs | PASS |
| s2_osc_baseline | 3.74e-05 | 0.001 abs | PASS |
| s3_exe_effect | 7.77e-05 | 0.001 abs | PASS |
| s3_exe_baseline | 7.43e-05 | 0.001 abs | PASS |
| s3_osc_baseline | 0.000228 | 0.001 abs | PASS |
| spin_baseline | 0 | 0.001 abs | PASS |

- new gpu baseline: IE 218.600, EA 103.535 kcal/mol (physical)
- new cpu baseline: IE 218.600, EA 103.535 kcal/mol (physical)

### SB_gas

Reference: `/data/PHO_WORK/sajagbe2/packages/emsuite/tests/integration_runs/opsins/campaigns/pose-restrain-2026-10-11/gas/SB/separate_singlet/results_SB_2026-10-11_06-35-42/SB_tuning_summary.csv` points [1, 30, 95, 111, 223, 252]  
Reference IE/EA baseline physical: **False**

**gpu vs reference: PASS** (38 columns)

| column | max diff | tolerance | status |
|---|---|---|---|
| cp_effect |  |  | SKIP (reference IE/EA baseline unphysical) |
| cp_baseline |  |  | SKIP (reference IE/EA baseline unphysical) |
| dm_baseline | 2.5e-12 | 0.001 abs | PASS |
| ea_effect |  |  | SKIP (reference IE/EA baseline unphysical) |
| ea_baseline |  |  | SKIP (reference IE/EA baseline unphysical) |
| efl_effect |  |  | SKIP (reference IE/EA baseline unphysical) |
| efl_baseline |  |  | SKIP (reference IE/EA baseline unphysical) |
| eng_effect |  |  | SKIP (reference IE/EA baseline unphysical) |
| eng_baseline |  |  | SKIP (reference IE/EA baseline unphysical) |
| gap_baseline | 7.37e-14 | 0.001 abs | PASS |
| gse_baseline | 2.33e-10 | 0.05 abs | PASS |
| hard_effect |  |  | SKIP (reference IE/EA baseline unphysical) |
| hard_baseline |  |  | SKIP (reference IE/EA baseline unphysical) |
| homo_baseline | 2.01e-13 | 0.001 abs | PASS |
| ie_effect |  |  | SKIP (reference IE/EA baseline unphysical) |
| ie_baseline |  |  | SKIP (reference IE/EA baseline unphysical) |
| lumo_baseline | 1.27e-13 | 0.001 abs | PASS |
| nfl_effect |  |  | SKIP (reference IE/EA baseline unphysical) |
| nfl_baseline |  |  | SKIP (reference IE/EA baseline unphysical) |
| s1_exe_effect | 3.06e-13 | 0.001 abs | PASS |
| s1_exe_baseline | 9.77e-14 | 0.001 abs | PASS |
| s1_osc_baseline | 4.23e-14 | 0.001 abs | PASS |
| s2_exe_effect | 2.9e-13 | 0.001 abs | PASS |
| s2_exe_baseline | 1.61e-13 | 0.001 abs | PASS |
| s2_osc_baseline | 5.85e-13 | 0.001 abs | PASS |
| s3_exe_effect | 1.92e-13 | 0.001 abs | PASS |
| s3_exe_baseline | 9.77e-15 | 0.001 abs | PASS |
| s3_osc_baseline | 7.66e-14 | 0.001 abs | PASS |
| spin_baseline | 0 | 0.001 abs | PASS |

**cpu vs reference: FAIL** (38 columns)

| column | max diff | tolerance | status |
|---|---|---|---|
| cp_effect |  |  | SKIP (reference IE/EA baseline unphysical) |
| cp_baseline |  |  | SKIP (reference IE/EA baseline unphysical) |
| dm_baseline | 4.24e-05 | 0.001 abs | PASS |
| ea_effect |  |  | SKIP (reference IE/EA baseline unphysical) |
| ea_baseline |  |  | SKIP (reference IE/EA baseline unphysical) |
| efl_effect |  |  | SKIP (reference IE/EA baseline unphysical) |
| efl_baseline |  |  | SKIP (reference IE/EA baseline unphysical) |
| eng_effect |  |  | SKIP (reference IE/EA baseline unphysical) |
| eng_baseline |  |  | SKIP (reference IE/EA baseline unphysical) |
| gap_baseline | 1.58e-06 | 0.001 abs | PASS |
| gse_baseline | 1.73e-07 | 0.05 abs | PASS |
| hard_effect |  |  | SKIP (reference IE/EA baseline unphysical) |
| hard_baseline |  |  | SKIP (reference IE/EA baseline unphysical) |
| homo_baseline | 3.22e-06 | 0.001 abs | PASS |
| ie_effect |  |  | SKIP (reference IE/EA baseline unphysical) |
| ie_baseline |  |  | SKIP (reference IE/EA baseline unphysical) |
| lumo_baseline | 4.79e-06 | 0.001 abs | PASS |
| nfl_effect |  |  | SKIP (reference IE/EA baseline unphysical) |
| nfl_baseline |  |  | SKIP (reference IE/EA baseline unphysical) |
| s1_exe_effect | 0.000158 | 0.001 abs | PASS |
| s1_exe_baseline | 3.06e-06 | 0.001 abs | PASS |
| s1_osc_baseline | 3.5e-07 | 0.001 abs | PASS |
| s2_exe_effect | 0.000109 | 0.001 abs | PASS |
| s2_exe_baseline | 5.25e-07 | 0.001 abs | PASS |
| s2_osc_effect | 0.00105 | 0.001 abs | FAIL |
| s2_osc_baseline | 3.98e-06 | 0.001 abs | PASS |
| s3_exe_effect | 2.31e-05 | 0.001 abs | PASS |
| s3_exe_baseline | 4.23e-07 | 0.001 abs | PASS |
| s3_osc_effect | 0.0011 | 0.001 abs | FAIL |
| s3_osc_baseline | 9.43e-06 | 0.001 abs | PASS |
| spin_baseline | 0 | 0.001 abs | PASS |

**cpu vs gpu: FAIL** (38 columns)

| column | max diff | tolerance | status |
|---|---|---|---|
| cp_baseline | 1.79e-07 | 0.05 abs | PASS |
| dm_baseline | 4.24e-05 | 0.001 abs | PASS |
| ea_baseline | 7.99e-09 | 0.05 abs | PASS |
| efl_baseline | 4e-09 | 0.002 abs | PASS |
| eng_baseline | 7.76e-09 | 0.002 abs | PASS |
| gap_baseline | 1.58e-06 | 0.001 abs | PASS |
| gse_baseline | 1.72e-07 | 0.05 abs | PASS |
| hard_baseline | 7.41e-09 | 0.002 abs | PASS |
| homo_baseline | 3.22e-06 | 0.001 abs | PASS |
| ie_effect | 1.5e-05 | 0.05 abs | PASS |
| ie_baseline | 3.5e-07 | 0.05 abs | PASS |
| lumo_baseline | 4.79e-06 | 0.001 abs | PASS |
| nfl_baseline | 2.6e-09 | 0.001 rel | PASS |
| s1_exe_effect | 0.000158 | 0.001 abs | PASS |
| s1_exe_baseline | 3.06e-06 | 0.001 abs | PASS |
| s1_osc_baseline | 3.5e-07 | 0.001 abs | PASS |
| s2_exe_effect | 0.000109 | 0.001 abs | PASS |
| s2_exe_baseline | 5.25e-07 | 0.001 abs | PASS |
| s2_osc_effect | 0.00105 | 0.001 abs | FAIL |
| s2_osc_baseline | 3.98e-06 | 0.001 abs | PASS |
| s3_exe_effect | 2.31e-05 | 0.001 abs | PASS |
| s3_exe_baseline | 4.23e-07 | 0.001 abs | PASS |
| s3_osc_effect | 0.0011 | 0.001 abs | FAIL |
| s3_osc_baseline | 9.43e-06 | 0.001 abs | PASS |
| spin_baseline | 0 | 0.001 abs | PASS |

- new gpu baseline: IE 161.261, EA -6.873 kcal/mol (physical)
- new cpu baseline: IE 161.261, EA -6.873 kcal/mol (physical)

### rPSB_nhexane

Reference: `/data/PHO_WORK/sajagbe2/packages/emsuite/tests/integration_runs/opsins/campaigns/pose-restrain-2026-10-11/solvents/rPSB/n-hexane/singlet/results_rPSB_2026-10-21_14-15-06/rPSB_tuning_summary.csv` points [57, 60, 126, 143, 163, 202]  
Reference IE/EA baseline physical: **False**

**gpu vs reference: PASS** (38 columns)

| column | max diff | tolerance | status |
|---|---|---|---|
| cp_effect |  |  | SKIP (reference IE/EA baseline unphysical) |
| cp_baseline |  |  | SKIP (reference IE/EA baseline unphysical) |
| dm_baseline | 5.28e-13 | 0.001 abs | PASS |
| ea_effect |  |  | SKIP (reference IE/EA baseline unphysical) |
| ea_baseline |  |  | SKIP (reference IE/EA baseline unphysical) |
| efl_effect |  |  | SKIP (reference IE/EA baseline unphysical) |
| efl_baseline |  |  | SKIP (reference IE/EA baseline unphysical) |
| eng_effect |  |  | SKIP (reference IE/EA baseline unphysical) |
| eng_baseline |  |  | SKIP (reference IE/EA baseline unphysical) |
| gap_baseline | 1.21e-13 | 0.001 abs | PASS |
| gse_baseline | 0 | 0.05 abs | PASS |
| hard_effect |  |  | SKIP (reference IE/EA baseline unphysical) |
| hard_baseline |  |  | SKIP (reference IE/EA baseline unphysical) |
| homo_baseline | 9.86e-14 | 0.001 abs | PASS |
| ie_effect |  |  | SKIP (reference IE/EA baseline unphysical) |
| ie_baseline |  |  | SKIP (reference IE/EA baseline unphysical) |
| lumo_baseline | 2.4e-14 | 0.001 abs | PASS |
| nfl_effect |  |  | SKIP (reference IE/EA baseline unphysical) |
| nfl_baseline |  |  | SKIP (reference IE/EA baseline unphysical) |
| s1_exe_effect | 2.05e-12 | 0.001 abs | PASS |
| s1_exe_baseline | 1.72e-13 | 0.001 abs | PASS |
| s1_osc_baseline | 8.26e-15 | 0.001 abs | PASS |
| s2_exe_effect | 1.43e-10 | 0.001 abs | PASS |
| s2_exe_baseline | 1.16e-13 | 0.001 abs | PASS |
| s2_osc_baseline | 9.73e-14 | 0.001 abs | PASS |
| s3_exe_effect | 6.94e-10 | 0.001 abs | PASS |
| s3_exe_baseline | 6.26e-14 | 0.001 abs | PASS |
| s3_osc_baseline | 9.24e-13 | 0.001 abs | PASS |
| spin_baseline | 0 | 0.001 abs | PASS |

- cpu vs reference: **MISSING results**

- cpu vs gpu: **MISSING results**

- new gpu baseline: IE 177.805, EA 85.943 kcal/mol (physical)

### rPSB_dmso

Reference: `/data/PHO_WORK/sajagbe2/packages/emsuite/tests/integration_runs/opsins/campaigns/pose-restrain-2026-10-11/solvents/rPSB/dimethylsulfoxide/singlet/results_rPSB_2026-10-23_13-31-59/rPSB_tuning_summary.csv` points [60, 75, 126, 143, 163, 202]  
Reference IE/EA baseline physical: **True**

**gpu vs reference: PASS** (38 columns)

| column | max diff | tolerance | status |
|---|---|---|---|
| cp_baseline | 1.43e-10 | 0.05 abs | PASS |
| dm_baseline | 8.51e-13 | 0.001 abs | PASS |
| ea_baseline | 2.85e-10 | 0.05 abs | PASS |
| efl_baseline | 6.26e-11 | 0.002 abs | PASS |
| eng_baseline | 6.19e-12 | 0.002 abs | PASS |
| gap_baseline | 9.95e-14 | 0.001 abs | PASS |
| gse_baseline | 5.82e-10 | 0.05 abs | PASS |
| hard_baseline | 1.86e-11 | 0.002 abs | PASS |
| homo_baseline | 5.15e-14 | 0.001 abs | PASS |
| ie_effect | 1.71e-09 | 0.05 abs | PASS |
| ie_baseline | 5.71e-10 | 0.05 abs | PASS |
| lumo_baseline | 4.71e-14 | 0.001 abs | PASS |
| nfl_baseline | 9.77e-12 | 0.001 rel | PASS |
| s1_exe_effect | 1.51e-13 | 0.001 abs | PASS |
| s1_exe_baseline | 9.33e-15 | 0.001 abs | PASS |
| s1_osc_baseline | 1.94e-15 | 0.001 abs | PASS |
| s2_exe_effect | 6.93e-13 | 0.001 abs | PASS |
| s2_exe_baseline | 1.51e-13 | 0.001 abs | PASS |
| s2_osc_baseline | 2.69e-13 | 0.001 abs | PASS |
| s3_exe_effect | 1.59e-13 | 0.001 abs | PASS |
| s3_exe_baseline | 3.42e-14 | 0.001 abs | PASS |
| s3_osc_baseline | 2.11e-13 | 0.001 abs | PASS |
| spin_baseline | 0 | 0.001 abs | PASS |

- cpu vs reference: **MISSING results**

- cpu vs gpu: **MISSING results**

- new gpu baseline: IE 134.048, EA 66.213 kcal/mol (physical)

### SB_water

Reference: `/data/PHO_WORK/sajagbe2/packages/emsuite/tests/integration_runs/opsins/campaigns/pose-restrain-2026-10-11/solvents/SB/water/singlet/results_SB_2026-10-26_09-21-49/SB_tuning_summary.csv` points [30, 54, 112, 128, 223, 252]  
Reference IE/EA baseline physical: **True**

- gpu vs reference: **MISSING results**

- cpu vs reference: **MISSING results**

- cpu vs gpu: **MISSING results**


## B. Baseline repeats (IE/EA drift)

| system | device | repeats ok | IE kcal/mol (min..max) | EA kcal/mol (min..max) | spread IE/EA | reference IE/EA | rejected states | verdict |
|---|---|---|---|---|---|---|---|---|
| rPSB_gas | gpu | 5/5 | 218.600..218.600 | 103.535..103.535 | 5.7e-10/8.6e-10 | 218.60/103.53 | 0 | **PASS** |
| rPSB_gas | cpu | 2/2 | 218.600..218.600 | 103.535..103.535 | 8.6e-10/1.4e-09 | 218.60/103.53 | 0 | **PASS** |
| SB_gas | gpu | 5/5 | 161.261..161.261 | -6.873..-6.873 | 8.6e-10/8.6e-10 | 161.26/-6.87 | 0 | **PASS** |
| SB_gas | cpu | 2/2 | 161.261..161.261 | -6.873..-6.873 | 2.9e-10/8.6e-10 | 161.26/-6.87 | 0 | **PASS** |
| rPSB_nhexane | gpu | 5/5 | 177.805..177.805 | 85.943..85.943 | 1.1e-09/1.1e-09 | -/- | 0 | **PASS** |
| rPSB_nhexane | cpu | 2/2 | 177.805..177.805 | 85.943..85.943 | 2e-09/2e-09 | -/- | 0 | **PASS** |
| rPSB_chloroform | gpu | 5/5 | 150.328..150.328 | 73.677..73.677 | 5.7e-10/1.1e-09 | -/- | 0 | **PASS** |
| rPSB_chloroform | cpu | 2/2 | 150.328..150.328 | 73.677..73.677 | 2.9e-10/1.1e-09 | -/- | 0 | **PASS** |
| rPSB_acetonitrile | gpu | 5/5 | 134.614..134.614 | 66.475..66.475 | 5.7e-10/5.7e-10 | 134.61/- | 0 | **PASS** |
| rPSB_acetonitrile | cpu | 2/2 | 134.614..134.614 | 66.475..66.475 | 5.7e-10/3.7e-09 | 134.61/- | 0 | **PASS** |
| SB_dmso | gpu | 5/5 | 125.511..125.511 | 32.421..32.421 | 5.7e-10/1.1e-09 | 125.51/- | 0 | **PASS** |
| SB_dmso | cpu | 2/2 | 125.511..125.511 | 32.421..32.421 | 1.7e-09/8.6e-10 | 125.51/- | 0 | **PASS** |

Reference match tolerance 0.5 kcal/mol; repeat spread tolerance 0.05 kcal/mol.

CPU vs GPU mean baseline difference (kcal/mol):

- rPSB_gas: IE -0.0000, EA -0.0000 → PASS
- SB_gas: IE +0.0000, EA +0.0000 → PASS
- rPSB_nhexane: IE -0.0000, EA +0.0000 → PASS
- rPSB_chloroform: IE -0.0000, EA +0.0000 → PASS
- rPSB_acetonitrile: IE -0.0000, EA +0.0000 → PASS
- SB_dmso: IE +0.0000, EA +0.0000 → PASS

