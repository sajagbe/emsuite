"""potential/resp.py: CPU RESP charges.

Reference values for water (examples/water-gpu/Water.xyz):
- B3LYP/6-31G* RESP from gpu4pyscf.pop.esp on a V100 (tests/probes/resp_cpu/water_charges.txt)
Ethanol two-stage HF/6-31G* reference: AmberTools 22 `resp` (stage 1 + stage 2 inputs
from `respgen`) run on the identical ESP points written by emsuite.potential.resp.esp_data.
"""

import numpy as np
import pytest
from pyscf import dft, gto, scf

from emsuite.potential.resp import (
    _shell_radius_bohr,
    equivalence_groups,
    resp_charges_from_xyz,
    resp_fit,
    resp_two_stage,
)

WATER = (
    "O 0.008935 0.404022 0.000000\nH -0.787313 -0.184699 0.000000\nH 0.778378 -0.219323 0.000000"
)


def test_resp_matches_gpu4pyscf_on_b3lyp_water():
    mol = gto.M(atom=WATER, basis="6-31g*", verbose=0)
    mf = dft.RKS(mol, xc="b3lyp").run()
    q = resp_fit(mol, mf.make_rdm1())
    gpu = np.array([-0.73270393, 0.36503842, 0.36766551])
    assert np.max(np.abs(q - gpu)) < 1e-6
    assert abs(q.sum()) < 1e-10


def test_hf_resp_charges_from_xyz(tmp_path):
    xyz = tmp_path / "water.xyz"
    xyz.write_text(f"3\nwater\n{WATER}\n")
    atoms, q = resp_charges_from_xyz(str(xyz))
    assert [a[0] for a in atoms] == ["O", "H", "H"]
    # two-stage: the symmetric hydrogens share one charge
    assert np.max(np.abs(q - np.array([-0.78839, 0.39420, 0.39420]))) < 1e-4
    assert abs(q.sum()) < 1e-10


def test_total_charge_is_respected(tmp_path):
    xyz = tmp_path / "h3o.xyz"
    xyz.write_text(
        "4\nhydronium\nO 0 0 0.1\nH 0.95 0 -0.2\nH -0.47 0.82 -0.2\nH -0.47 -0.82 -0.2\n"
    )
    _, q = resp_charges_from_xyz(str(xyz), charge=1)
    assert abs(q.sum() - 1.0) < 1e-10


@pytest.mark.parametrize("symbol,angstrom", [("C", 1.50), ("Cl", 1.70), ("Br", 1.85), ("I", 1.98)])
def test_fitting_shell_radii(symbol, angstrom):
    # Merz-Kollman up to Cl, then the emsuite (vdw-surfgen) table.
    assert abs(_shell_radius_bohr(symbol) * 0.52917721092 - angstrom) < 1e-6


ETHANOL = """C -0.863160 0.220412 0.202725
C 0.406289 -0.375004 -0.369994
O 1.527382 0.403609 0.022204
H -1.738486 -0.358822 -0.104136
H -0.988335 1.255433 -0.131901
H -0.821112 0.242401 1.296668
H 0.551466 -1.401287 -0.019347
H 0.364532 -0.386510 -1.463002
H 1.561426 0.399768 0.994021"""


def test_two_stage_groups_match_respgen_for_ethanol():
    mol = gto.M(atom=ETHANOL, basis="6-31g*", verbose=0)
    stage1, stage2, refit = equivalence_groups(mol)
    assert stage1 == []
    assert sorted(stage2) == [[3, 4, 5], [6, 7]]  # CH3 and CH2 hydrogens
    assert refit == [0, 1, 3, 4, 5, 6, 7]  # O and hydroxyl H keep stage-1 charges


def test_two_stage_matches_amber_resp_for_ethanol():
    mol = gto.M(atom=ETHANOL, basis="6-31g*", verbose=0)
    q = resp_two_stage(mol, scf.RHF(mol).run().make_rdm1())
    amber = np.array(
        [
            -0.129099,
            0.370818,
            -0.704267,
            0.024490,
            0.024490,
            0.024490,
            -0.015954,
            -0.015954,
            0.420986,
        ]
    )
    assert np.max(np.abs(q - amber)) < 1e-5
