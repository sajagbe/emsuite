"""Collapsed ion SCF solutions must not become IE/EA baselines."""

from emsuite.core.molecule import _unphysical_reason


class _MF:
    def __init__(self, s2):
        self._s2 = s2

    def spin_square(self):
        return (self._s2, 0.0)


def test_rejects_energy_far_from_reference():
    # rPSB n-hexane run: neutral -873.95 Ha, anion picked at -1291.75 Ha
    assert _unphysical_reason(_MF(3.76), 3, -1291.754, reference_energy=-873.949)


def test_rejects_spin_contaminated_open_shell():
    assert _unphysical_reason(_MF(2.53), 1, -1081.2)  # doublet, expected 0.75
    assert _unphysical_reason(_MF(6.26), 3, -1291.75)  # quartet, expected 3.75


def test_accepts_physical_ion():
    # rPSB good run: anion doublet -874.114 Ha, <S^2> 0.79
    assert _unphysical_reason(_MF(0.79), 1, -874.114, reference_energy=-873.949) is None
    assert _unphysical_reason(None, 0, -874.0, reference_energy=-873.949) is None


def test_cpu_objects_are_not_rechecked():
    from pyscf import gto, scf

    from emsuite.core.molecule import _cpu_energy_mismatch

    mf = scf.RHF(gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)).run()
    assert _cpu_energy_mismatch(mf, mf.e_tot) is None


def test_missing_state_raises_clear_error(monkeypatch, tmp_path):
    import pytest

    from emsuite.tuning import runner

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "create_alone_molecule_objects", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="No trustworthy SCF solution for the neutral"):
        runner.create_molecule_objects(
            "x.xyz", "sto-3g", 0, "dft", "b3lyp", 0, False, {"neutral": True}, 1, False
        )
