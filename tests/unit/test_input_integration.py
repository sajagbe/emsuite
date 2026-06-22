"""Parse surface.in using the shared config module (replaces exec)."""

from pathlib import Path

import pytest

from emsuite.surface import parse_surface_input


def test_parse_surface_input_from_example():
    example = Path("examples/tuning/CCO2-exe/surface.in")
    if not example.exists():
        pytest.skip("example surface.in not present")
    params = parse_surface_input(str(example))
    assert params["input_type"] == "SMILES"
    assert params["input_data"] == "CCO"
    assert params["output_surf"] == "CCO2.surf"


def test_get_tuning_parameters_from_example():
    from emsuite.tuning import get_tuning_parameters

    example = Path("examples/tuning/CCO2-exe/tuning.in")
    if not example.exists():
        pytest.skip("example tuning.in not present")
    params = get_tuning_parameters(str(example))
    assert params["molecule"] == "CCO_opt2.xyz"
    assert params["surface_file"] == "CCO2.surf"
    assert "exe" in params["properties"]
