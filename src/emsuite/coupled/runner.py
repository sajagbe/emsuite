"""Coupled potential → tuning pipeline (parse helpers only; execute via CoupledInput.run)."""

from __future__ import annotations

from emsuite.config import parse_config_file
from emsuite.config.schemas import validate_coupled_params


def parse_coupled_input(input_file: str) -> dict:
    from emsuite.inputs import CoupledInput, _channel_defaults

    params = parse_config_file(input_file, defaults=_channel_defaults(CoupledInput))
    return validate_coupled_params(params)
