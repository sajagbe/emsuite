"""Configured GPU IDs must not override failed CUDA detection."""

import pytest

from emsuite.inputs import TuningInput
from emsuite.tuning import runner


@pytest.mark.parametrize("usable, expected_gpus", [(0, []), (1, ["4", "5", "6"])])
def test_hardware_check_controls_baseline_gpu_use(tmp_path, monkeypatch, usable, expected_gpus):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ligand.xyz").write_text("1\nligand\nH 0 0 0\n")
    (tmp_path / "ligand.surf").write_text("x y z q\n1.2 0 0 0.1\n")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6")
    monkeypatch.setattr(runner.core, "CUPY_AVAILABLE", True)
    monkeypatch.setattr(runner.core, "check_cpu_info", lambda: 1)

    def check_gpu():
        # CUDA detection must happen after pinning, even when detection fails.
        import os

        assert os.environ["CUDA_VISIBLE_DEVICES"] == "4"
        return usable

    monkeypatch.setattr(runner.core, "check_gpu_info", check_gpu)

    class ReachedSCF(Exception):
        pass

    def check_scf_args(*args, **kwargs):
        assert args[6] is bool(usable)  # gpu_available
        assert kwargs["gpus"] == expected_gpus
        raise ReachedSCF

    monkeypatch.setattr(runner, "create_molecule_objects", check_scf_args)
    with pytest.raises(ReachedSCF):
        runner._run_tuning(
            TuningInput(molecule="ligand.xyz", surface_file="ligand.surf", properties=("gse",))
        )
