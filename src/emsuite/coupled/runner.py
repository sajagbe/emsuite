"""Coupled potential → tuning pipeline (parse helpers only; execute via CoupledInput.run)."""

from __future__ import annotations

from pathlib import Path

from emsuite.config import parse_assignments, parse_config_file
from emsuite.config.schemas import reject_removed_coupled_keys, validate_coupled_params


def parse_coupled_input(input_file: str) -> dict:
    from emsuite.inputs import CoupledInput, _channel_defaults

    path = Path(input_file)
    if path.exists():
        reject_removed_coupled_keys(parse_assignments(path.read_text()))
    params = parse_config_file(input_file, defaults=_channel_defaults(CoupledInput))
    return validate_coupled_params(params)
