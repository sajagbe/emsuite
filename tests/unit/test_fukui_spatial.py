"""Fukui spatial projection tests."""

import numpy as np
from pyscf import gto, scf

from emsuite.tuning.properties.fukui_spatial import mulliken_charges, project_atom_property_to_point


def test_project_nearest_atom():
    atoms = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    values = np.array([1.0, 2.0])
    point = np.array([0.1, 0.0, 0.0])
    assert project_atom_property_to_point(atoms, values, point) == 1.0


def test_mulliken_charges_one_per_atom():
    mol = gto.M(atom="C 0 0 0; H 1 0 0", basis="sto-3g")
    mf = scf.RHF(mol).run()
    charges = mulliken_charges(mf)
    assert charges.shape == (2,)
