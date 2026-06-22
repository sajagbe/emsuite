"""Unit tests for emsuite.config."""

from pathlib import Path

from emsuite.config import parse_assignments, parse_config_file


def test_parse_assignments_basic():
    content = """
    # comment
    molecule = 'water.xyz'
    properties = ['homo', 'lumo']
    parallel = True
    num_procs = None
    """
    params = parse_assignments(content)
    assert params["molecule"] == "water.xyz"
    assert params["properties"] == ["homo", "lumo"]
    assert params["parallel"] is True
    assert params["num_procs"] is None


def test_parse_assignments_unquoted_string():
    params = parse_assignments("functional = b3lyp\n")
    assert params["functional"] == "b3lyp"


def test_parse_config_file_with_defaults(tmp_path: Path):
    config = tmp_path / "surface.in"
    config.write_text("input_type = 'SMILES'\ninput_data = 'CCO'\nsurface_charge = 0.1\n")
    defaults = {
        "input_type": None,
        "input_data": None,
        "surface_charge": 0.10,
        "surface_density": 1.0,
    }
    params = parse_config_file(config, defaults=defaults)
    assert params["input_type"] == "SMILES"
    assert params["input_data"] == "CCO"
    assert params["surface_charge"] == 0.1
    assert params["surface_density"] == 1.0


def test_parse_config_file_tuning_style(tmp_path: Path):
    config = tmp_path / "tuning.in"
    config.write_text("molecule = 'CCO_opt.xyz'\nsurface_file = 'CCO.surf'\n")
    params = parse_config_file(config)
    assert params["molecule"] == "CCO_opt.xyz"
    assert params["surface_file"] == "CCO.surf"


def test_parse_config_file_missing_returns_empty_or_defaults(tmp_path: Path):
    missing = tmp_path / "missing.in"
    assert parse_config_file(missing) == {}
    defaults = {"a": 1}
    assert parse_config_file(missing, defaults=defaults) == {"a": 1}
