"""Office quote Easter egg printed at successful tuning completion."""

from __future__ import annotations

import json
from unittest.mock import patch

from emsuite.core.hardware import print_office_quote


def test_print_office_quote_formats_response(capsys):
    payload = json.dumps(
        {"quote": "Identity theft is not a joke, Jim!", "character": "Dwight Schrute"}
    ).encode()

    class _Http:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return payload

    with patch("emsuite.core.hardware.urllib.request.urlopen", return_value=_Http()):
        print_office_quote()

    out = capsys.readouterr().out
    assert "Fetching inspirational quote..." in out
    assert "Identity theft is not a joke, Jim!" in out
    assert "Dwight Schrute" in out


def test_print_office_quote_swallows_network_errors(capsys):
    with patch(
        "emsuite.core.hardware.urllib.request.urlopen",
        side_effect=TimeoutError("timed out"),
    ):
        print_office_quote()  # must not raise

    out = capsys.readouterr().out
    assert "Fetching inspirational quote..." in out
    assert "Could not fetch Office quote" in out
