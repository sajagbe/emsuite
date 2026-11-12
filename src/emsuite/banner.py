"""One shared, background-free logo and introduction per calculation job."""

import os
import sys
from contextvars import ContextVar
from functools import wraps

from ._logo import LOGO_ASCII, RGB_ROWS

_active_job = ContextVar("emsuite_banner_active", default=False)


def _color_enabled():
    if "NO_COLOR" in os.environ:
        return False
    if os.environ.get("FORCE_COLOR", "0") != "0":
        return True
    if not getattr(sys.stdout, "isatty", lambda: False)() or os.environ.get("TERM") == "dumb":
        return False
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            kernel = ctypes.windll.kernel32
            kernel.GetStdHandle.argtypes = [wintypes.DWORD]
            kernel.GetStdHandle.restype = wintypes.HANDLE
            kernel.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            kernel.GetConsoleMode.restype = wintypes.BOOL
            kernel.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            kernel.SetConsoleMode.restype = wintypes.BOOL
            mode = wintypes.DWORD()
            handle = kernel.GetStdHandle(-11)
            if not kernel.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            return bool(kernel.SetConsoleMode(handle, mode.value | 0x0004))
        except (AttributeError, OSError):
            return False
    return True


def render_logo(color=False):
    """Static ASCII art with original foreground RGB colors; no background codes."""
    if not color:
        return LOGO_ASCII
    lines = []
    for line, palette in zip(LOGO_ASCII.splitlines(), RGB_ROWS, strict=True):
        rendered, previous = [], None
        for i, char in enumerate(line):
            rgb = palette[i * 6 : i * 6 + 6]
            if char != " " and rgb != previous:
                red, green, blue = (int(rgb[j : j + 2], 16) for j in (0, 2, 4))
                rendered.append(f"\033[38;2;{red};{green};{blue}m")
                previous = rgb
            rendered.append(char)
        lines.append("".join(rendered) + "\033[39m")
    return "\n".join(lines)


def print_startup_message():
    from emsuite import __version__

    color = _color_enabled()
    print("\n" + render_logo(color))
    name = "EMSuite"
    if color:
        palette = (
            (65, 109, 255),
            (90, 140, 255),
            (145, 180, 255),
            (245, 245, 255),
            (255, 176, 176),
            (255, 110, 110),
            (255, 65, 65),
        )
        name = (
            "".join(
                f"\033[38;2;{r};{g};{b}m{letter}"
                for letter, (r, g, b) in zip(name, palette, strict=True)
            )
            + "\033[39m"
        )
    print("=" * 64)
    print(f"{name} {__version__} - Electrostatic Map Suite")
    print("By Stephen O. Ajagbe")
    print("Electrostatic surfaces, potential fields and tuning maps.")
    print("Built on efforts by the Gozem Lab")
    print("See: https://pubs.acs.org/doi/10.1021/acs.jpcb.9b00489")
    print("=" * 64 + "\n")


def with_banner(function):
    """CLI and nested channel calls share a context; separate jobs print anew."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        if _active_job.get():
            return function(*args, **kwargs)
        token = _active_job.set(True)
        try:
            print_startup_message()
            return function(*args, **kwargs)
        finally:
            _active_job.reset(token)

    return wrapped
