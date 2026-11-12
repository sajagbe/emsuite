"""Automatic RESP availability/fallback and allocation regression checks."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from emsuite.core import _gpu
from emsuite.inputs import CoupledInput
from emsuite.potential import resp
from emsuite.potential.spec import PotentialSpec


def test_no_cupy_runs_cpu_and_records_diagnostics(monkeypatch, capsys):
    monkeypatch.setattr(_gpu, "CUPY_AVAILABLE", False)
    diagnostics = {}
    q = resp.resp_charges_from_atoms(
        [("O", 0.0, 0.0, 0.0), ("H", 0.0, 0.757, 0.587), ("H", 0.0, -0.757, 0.587)],
        diagnostics=diagnostics,
    )
    assert isinstance(q, np.ndarray) and abs(q.sum()) < 1e-10
    assert diagnostics["backend"] == "cpu"
    assert {"scf_s", "esp_and_normal_equations_s", "two_stage_fit_s", "total_s"} <= diagnostics[
        "timings_seconds"
    ].keys()
    assert "CuPy is not installed" in capsys.readouterr().out


def test_cupy_import_with_unusable_driver_is_not_gpu_availability(monkeypatch):
    def no_driver():
        raise RuntimeError("CUDA driver unavailable")

    monkeypatch.setattr(_gpu, "CUPY_AVAILABLE", True)
    monkeypatch.setattr(
        _gpu,
        "cp",
        SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=no_driver))),
    )
    selected, reason = resp._gpu_backend()
    assert selected is None and "CUDA driver unavailable" in reason


def test_gpu_selection_pins_then_restores_allocation(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")
    cp = SimpleNamespace(cuda=SimpleNamespace(Device=lambda _: nullcontext()))

    def available():
        import os

        assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
        return (cp, "GPU SCF", "GPU DFT"), None

    monkeypatch.setattr(resp, "_gpu_backend", available)
    with resp._backend("auto") as selected:
        assert selected == (cp, "GPU SCF", "GPU DFT")
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,5"


def test_numerical_failure_is_not_silently_retried_on_cpu(monkeypatch):
    cp = SimpleNamespace(cuda=SimpleNamespace(Device=lambda _: nullcontext()))
    monkeypatch.setattr(resp, "_gpu_backend", lambda: ((cp, object(), object()), None))
    with pytest.raises(RuntimeError, match="numerical SCF failure"):
        with resp._backend("auto"):
            raise RuntimeError("numerical SCF failure")


def test_forced_gpu_cannot_pass_by_falling_back(monkeypatch):
    monkeypatch.setattr(resp, "_gpu_backend", lambda: (None, "no allocated GPU"))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError, match="GPU RESP requested but unavailable"):
        with resp._backend("gpu"):
            pytest.fail("unavailable GPU must not enter calculation")


def test_new_apbs_default_and_explicit_legacy_override():
    for spacing in (0.125, None):
        coupled = CoupledInput(molecule="m.xyz", pdb="p.pdb", grid_spacing=spacing)
        assert coupled.to_potential_input().grid_spacing == spacing
    assert CoupledInput(molecule="m.xyz", pdb="p.pdb").grid_spacing == 0.125
    assert PotentialSpec(pdb="p.pdb", surface_file="s.surf").grid_spacing == 0.125
