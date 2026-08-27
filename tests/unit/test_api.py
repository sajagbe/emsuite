"""Unit tests for the keyword-argument API and config resolution."""

import numpy as np
import pytest

from emsuite import api
from emsuite.config import UNSET, resolve_config
from emsuite.inputs import CoupledInput, PotentialInput, SurfaceInput, TuningInput
from emsuite.results import CoupledResult, PotentialResult, SurfaceResult, TuningResult


def test_resolve_precedence_defaults_config_overrides():
    defaults = {"a": 1, "b": 2, "c": 3}
    config = {"b": 20, "c": 30}
    overrides = {"c": 300}
    out = resolve_config(config, overrides, defaults=defaults)
    assert out == {"a": 1, "b": 20, "c": 300}


def test_resolve_ignores_unset_overrides():
    out = resolve_config(None, {"x": UNSET, "y": 5}, defaults={"x": 1})
    assert out == {"x": 1, "y": 5}


def test_resolve_dict_config_passthrough():
    assert resolve_config({"k": "v"}) == {"k": "v"}


def test_resolve_file_config(tmp_path):
    cfg = tmp_path / "c.in"
    cfg.write_text("molecule = 'm.xyz'\nproperties = ['homo']\n")
    out = resolve_config(str(cfg))
    assert out["molecule"] == "m.xyz"
    assert out["properties"] == ["homo"]


def test_resolve_rejects_bad_type():
    with pytest.raises(TypeError):
        resolve_config(config=42)


def test_tune_passes_overrides_to_input(monkeypatch):
    captured = {}

    def fake_run(self):
        captured["molecule"] = self.molecule
        captured["surface_file"] = self.surface_file
        captured["properties"] = self.properties
        return TuningResult(results_dir="ok")

    monkeypatch.setattr(TuningInput, "run", fake_run)
    result = api.tune(molecule="m.xyz", surface_file="s.surf", properties=["gap"])
    assert result.results_dir == "ok"
    assert captured["molecule"] == "m.xyz"
    assert captured["surface_file"] == "s.surf"
    assert captured["properties"] == ("gap",)


def test_tune_explicit_kwarg_overrides_config(monkeypatch):
    captured = {}

    def fake_run(self):
        captured["molecule"] = self.molecule
        captured["parallel"] = self.parallel
        return TuningResult()

    monkeypatch.setattr(TuningInput, "run", fake_run)
    api.tune(config={"molecule": "from_config.xyz", "surface_file": "s.surf", "parallel": True}, parallel=False)
    assert captured["molecule"] == "from_config.xyz"
    assert captured["parallel"] is False


def test_potential_kwargs_to_input(monkeypatch):
    def fake_run(self):
        assert self.molecule == "m.xyz"
        assert self.quantity == "charge"
        return PotentialResult(
            coords=np.zeros((1, 3)), values=np.zeros(1), quantity="charge", path="p.surf"
        )

    monkeypatch.setattr(PotentialInput, "run", fake_run)
    assert api.potential(molecule="m.xyz", quantity="charge") == "p.surf"


def test_surface_kwargs_to_input(monkeypatch):
    def fake_run(self):
        assert self.input_type == "SMILES"
        assert self.input_data == "CCO"
        return SurfaceResult(coords=np.zeros((1, 3)), values=np.zeros(1), path="x.surf")

    monkeypatch.setattr(SurfaceInput, "run", fake_run)
    assert api.surface(input_type="SMILES", input_data="CCO") == "x.surf"


def test_coupled_kwargs_to_input(monkeypatch):
    def fake_run(self):
        assert self.molecule == "m.xyz"
        assert self.properties == ("homo", "lumo")
        return CoupledResult(
            potential=PotentialResult(
                coords=np.zeros((1, 3)), values=np.zeros(1), quantity="charge", path="c.surf"
            ),
            tuning=TuningResult(),
        )

    monkeypatch.setattr(CoupledInput, "run", fake_run)
    api.coupled(molecule="m.xyz", properties=["homo", "lumo"])


def test_top_level_tune_exported():
    import emsuite

    assert emsuite.tune is api.tune
    assert emsuite.api is api
