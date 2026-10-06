"""Potential channel input parsing."""

from __future__ import annotations

from emsuite.config import parse_config_file
from emsuite.config.schemas import validate_potential_params


def parse_potential_input(input_file: str) -> dict:
    from emsuite.inputs import PotentialInput, _channel_defaults

    params = parse_config_file(input_file, defaults=_channel_defaults(PotentialInput))
    return validate_potential_params(params)
