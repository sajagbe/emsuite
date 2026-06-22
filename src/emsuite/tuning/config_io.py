"""Tuning input file parsing."""

from emsuite.config import parse_config_file


def get_tuning_parameters(filepath="tuning.in"):
    """
    Search for tuning.in file and return parameters.

    Args:
        filepath (str): Path to tuning file, defaults to 'tuning.in'

    Returns:
        dict: Dictionary of tuning parameters
    """
    try:
        return parse_config_file(filepath)
    except OSError as e:
        print(f"Error parsing tuning.in file: {e}")
        return {}


