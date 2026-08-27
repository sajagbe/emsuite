"""Unit tests for the keyword-argument API and config resolution."""

import pytest

from emsuite import api
from emsuite.config import UNSET, resolve_config


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


def test_tune_passes_overrides_to_runner(monkeypatch):
    captured = {}
    monkeypatch.setattr(api, "_run_tuning", lambda params: captured.update(params) or "ok")
    result = api.tune(molecule="m.xyz", surface_file="s.surf", properties=["gap"])
    assert result == "ok"
    assert captured["molecule"] == "m.xyz"
    assert captured["surface_file"] == "s.surf"
    assert captured["properties"] == ["gap"]
    # Unspecified kwargs must not leak through as keys.
    assert "solvent" not in captured


def test_tune_explicit_kwarg_overrides_config(monkeypatch):
    captured = {}
    monkeypatch.setattr(api, "_run_tuning", lambda params: captured.update(params))
    api.tune(config={"molecule": "from_config.xyz", "parallel": True}, parallel=False)
    assert captured["molecule"] == "from_config.xyz"
    assert captured["parallel"] is False


def test_potential_kwargs_to_runner(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        api, "run_potential_calculation", lambda params: captured.update(params) or "p.surf"
    )
    assert api.potential(molecule="m.xyz", bond_scan_atoms=[0, 1]) == "p.surf"
    assert captured["molecule"] == "m.xyz"
    assert captured["bond_scan_atoms"] == [0, 1]


def test_surface_kwargs_to_runner(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        api, "run_surface_calculation", lambda params: captured.update(params) or "x.surf"
    )
    assert api.surface(input_type="SMILES", input_data="CCO") == "x.surf"
    assert captured == {"input_type": "SMILES", "input_data": "CCO"}


def test_coupled_kwargs_to_runner(monkeypatch):
    captured = {}
    monkeypatch.setattr(api, "run_coupled_calculation", lambda params: captured.update(params))
    api.coupled(molecule="m.xyz", properties=["homo", "lumo"])
    assert captured["molecule"] == "m.xyz"
    assert captured["properties"] == ["homo", "lumo"]


def test_top_level_tune_exported():
    import emsuite

    assert emsuite.tune is api.tune
    assert emsuite.api is api
