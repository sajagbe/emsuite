"""Branding is shared across channels, nested once, with foreground-only color."""

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

from emsuite import banner


@pytest.mark.parametrize("flag", ["-s", "-t", "-p", "-c"])
def test_each_cli_channel_prints_one_banner_before_channel(tmp_path, monkeypatch, capsys, flag):
    from emsuite.cli import main as cli

    inp = tmp_path / "job.in"
    inp.write_text("# test input\n")

    @banner.with_banner
    def nested_stage():
        print("CHANNEL BODY")

    class FakeInput:
        @classmethod
        def from_file(cls, path):
            assert path == inp
            return cls()

        def run(self):
            nested_stage()
            nested_stage()

    for name in ("SurfaceInput", "TuningInput", "PotentialInput", "CoupledInput"):
        monkeypatch.setattr(cli, name, FakeInput)
    monkeypatch.setenv("NO_COLOR", "1")
    cli.main([flag, str(inp)])
    text = capsys.readouterr().out
    assert text.count("EMSuite ") == 1
    assert text.count("Built on efforts by the Gozem Lab") == 1
    assert text.index(banner.LOGO_ASCII) < text.index("EMSuite ") < text.index("CHANNEL BODY")


def test_separate_jobs_and_failures_reset_banner_context(capsys):
    @banner.with_banner
    def run(fail=False):
        if fail:
            raise ValueError("test failure")

    with pytest.raises(ValueError):
        run(True)
    run()
    assert capsys.readouterr().out.count("Electrostatic Map Suite") == 2


def test_colored_logo_preserves_ascii_and_has_no_background_codes(monkeypatch, capsys):
    colored = banner.render_logo(True)
    assert re.sub(r"\x1b\[[0-9;]+m", "", colored) == banner.LOGO_ASCII
    codes = re.findall(r"\x1b\[([0-9;]+)m", colored)
    assert codes and all(code == "39" or code.startswith("38;2;") for code in codes)
    rgbs = [tuple(map(int, code.split(";")[2:])) for code in codes if code.startswith("38;2;")]
    assert any(b > r + 50 for r, g, b in rgbs)  # map blue
    assert any(r > b + 50 for r, g, b in rgbs)  # map red
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    banner.print_startup_message()
    plain = re.sub(r"\x1b\[[0-9;]+m", "", capsys.readouterr().out)
    assert "EMSuite " in plain


def test_no_color_respects_plain_log_output(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert not banner._color_enabled()
    assert "\x1b" not in banner.render_logo(False)


def test_windows_console_enables_ansi_without_resetting_existing_mode(monkeypatch):
    import ctypes

    modes = []

    def get_mode(handle, ptr):
        ptr._obj.value = 0x0002
        return 1

    kernel = SimpleNamespace(
        GetStdHandle=lambda _: 11,
        GetConsoleMode=get_mode,
        SetConsoleMode=lambda handle, mode: modes.append(mode) or 1,
    )
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(kernel32=kernel), raising=False)
    monkeypatch.setattr(banner, "os", SimpleNamespace(name="nt", environ={}))
    monkeypatch.setattr(banner, "sys", SimpleNamespace(stdout=SimpleNamespace(isatty=lambda: True)))
    assert banner._color_enabled()
    assert modes == [0x0006]


def test_readme_logo_variants_have_no_canvas_background():
    root = Path(__file__).resolve().parents[2]
    for theme in ("light", "dark"):
        svg = ET.parse(root / f"assets/emsuite-logo-{theme}.svg").getroot()
        assert not any(child.tag.endswith("}rect") for child in svg)
