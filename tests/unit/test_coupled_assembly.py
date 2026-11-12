"""Coupled stage ordering, units, failure preservation and reuse contracts."""

import json
from pathlib import Path

import numpy as np
import pytest

from emsuite.inputs import CoupledInput, PotentialInput, SurfaceInput, TuningInput
from emsuite.results import PotentialResult, SurfaceResult, TuningResult
from emsuite.surface.io import load_surf, save_surf, surface_quantity


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "m.xyz").write_text("1\nfixed oxygen\nO 10 -3 2\n")
    (tmp_path / "p.pdb").write_text("PDB handled by potential stage")
    return dict(molecule="m.xyz", pdb="p.pdb", properties=["homo"])


def install_stages(monkeypatch, fail=None):
    calls = []
    points = np.array([[11.52, -3.0, 2.0], [8.48, -3.0, 2.0], [10.0, -1.48, 2.0]])

    def surface(spec):
        calls.append("surface")
        assert spec.optimize is False
        assert Path(spec.input_data).read_text().endswith("O 10 -3 2\n")
        save_surf(points, 0.0, spec.output_surf)
        return SurfaceResult.from_surf(spec.output_surf)

    def potential(spec):
        calls.append("potential")
        if fail == "potential":
            raise RuntimeError("APBS failure")
        assert spec.quantity == "charge"
        coords, _ = load_surf(spec.surface_file)
        save_surf(coords, [0.1, -0.2, 0.3], spec.output_surf, quantity="charge")
        return PotentialResult.from_surf(spec.output_surf, quantity="charge")

    def tuning(spec):
        calls.append("tuning")
        np.testing.assert_allclose(load_surf(spec.surface_file)[1], [0.1, -0.2, 0.3])
        assert surface_quantity(spec.surface_file) == "charge"
        directory = Path.cwd() / "results_m_test"
        directory.mkdir()
        # The production tuning runner changes cwd before returning.
        monkeypatch.chdir(directory)
        return TuningResult(results_dir=directory.name)

    monkeypatch.setattr(SurfaceInput, "run", surface)
    monkeypatch.setattr(PotentialInput, "run", potential)
    monkeypatch.setattr(TuningInput, "run", tuning)
    return calls


def test_generate_then_reuse_same_surface(inputs, tmp_path, monkeypatch):
    calls = install_stages(monkeypatch)
    original = (tmp_path / "m.xyz").read_bytes()
    first = CoupledInput(**inputs).run()
    assert calls == ["surface", "potential", "tuning"]
    assert Path.cwd() == tmp_path
    assert (tmp_path / "m.xyz").read_bytes() == original
    assert Path(first.tuning.results_dir).is_dir()
    manifest = json.loads((Path(first.run_dir) / "run.json").read_text())
    assert manifest["status"] == "completed" and manifest["charge_units"] == "e"
    calls.clear()
    second = CoupledInput(**inputs, surface_file=first.surface.path).run()
    assert calls == ["potential", "tuning"]
    np.testing.assert_array_equal(first.potential.values, second.potential.values)
    assert first.run_dir != second.run_dir


def test_reuse_charges_skips_both_stages(inputs, monkeypatch):
    calls = install_stages(monkeypatch)
    first = CoupledInput(**inputs).run()
    calls.clear()
    second = CoupledInput(molecule=inputs["molecule"], potential_surf=first.potential.path).run()
    assert calls == ["tuning"] and second.surface is None


def test_failure_stops_pipeline_and_preserves_inputs(inputs, tmp_path, monkeypatch):
    calls = install_stages(monkeypatch, fail="potential")
    with pytest.raises(RuntimeError, match="APBS failure"):
        CoupledInput(**inputs, output_dir="failed_run").run()
    assert calls == ["surface", "potential"] and Path.cwd() == tmp_path
    manifest = json.loads((tmp_path / "failed_run/run.json").read_text())
    assert manifest["status"] == "failed" and manifest["stage"] == "potential"
    assert Path(manifest["inputs"]["molecule"]["snapshot"]).is_file()


@pytest.mark.parametrize("reused", [False, True])
def test_rejects_potential_quantity_before_running(inputs, reused):
    with pytest.raises(ValueError, match="requires charges"):
        CoupledInput(
            **inputs, potential_quantity="potential", potential_surf="old.surf" if reused else None
        )


def test_typed_potential_file_is_not_charge_input(inputs, tmp_path):
    surface = tmp_path / "potential.surf"
    save_surf([[11.52, -3.0, 2.0]], 1.0, surface, quantity="potential")
    with pytest.raises(ValueError, match="requires charge"):
        CoupledInput(molecule=inputs["molecule"], potential_surf=str(surface)).run()
    assert not list(tmp_path.glob("coupled_*"))


def test_invalid_reused_frame_rejected_before_potential(inputs, tmp_path):
    surface = tmp_path / "wrong.surf"
    save_surf([[0, 0, 0], [10, 0, 0], [20, 0, 0]], 0.0, surface)
    with pytest.raises(ValueError, match="coordinate frame"):
        CoupledInput(**inputs, surface_file=str(surface)).run()


def test_file_can_request_automatic_surface(inputs, tmp_path):
    cfg = tmp_path / "coupled.in"
    cfg.write_text(
        "molecule = 'm.xyz'\npdb = 'p.pdb'\nsurface_density = 0.01\nsurface_scale = 1.2\n"
    )
    spec = CoupledInput.from_file(cfg)
    assert spec.surface_file is None and spec.surface_scale == 1.2
    assert spec.surface_density == 0.01
