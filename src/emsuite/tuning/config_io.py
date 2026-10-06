"""Tuning input file parsing."""

from __future__ import annotations

from emsuite.config import parse_config_file
from emsuite.config.schemas import validate_tuning_params


def parse_tuning_input(input_file: str = "tuning.in") -> dict:
    """Parse a tuning.in file and return validated parameters."""
    from emsuite.inputs import TuningInput, _channel_defaults

    try:
        params = parse_config_file(input_file, defaults=_channel_defaults(TuningInput))
        return validate_tuning_params(params)
    except OSError as e:
        print(f"Error parsing tuning.in file: {e}")
        return {}
