"""Regenerate the bundled ASCII/RGB logo; CairoSVG/Pillow are build tools only."""

import io
from pathlib import Path

import cairosvg
from PIL import Image


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "assets/emsuite-logo-light.svg"
    png = cairosvg.svg2png(url=str(source), output_width=440)
    image = Image.open(io.BytesIO(png)).convert("RGBA")
    width = 64
    height = round(image.height / image.width * width * 0.45)
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    ramp = " .:-=+*#%@"
    lines, colors = [], []
    for y in range(height):
        line, row = "", []
        for x in range(width):
            red, green, blue, alpha = image.getpixel((x, y))
            luminance = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255
            ink = alpha / 255 * (0.3 + 0.7 * (1 - luminance))
            line += ramp[min(len(ramp) - 1, round(ink * (len(ramp) - 1)))]
            row.append(f"{red:02x}{green:02x}{blue:02x}")
        line = line.rstrip()
        lines.append(line)
        colors.append("".join(row[: len(line)]))
    while lines and not lines[0]:
        lines.pop(0)
        colors.pop(0)
    while lines and not lines[-1]:
        lines.pop()
        colors.pop()
    art = "\n".join(lines)
    text = (
        '"""Generated from assets/emsuite-logo-light.svg by scripts/build_logo_art.py."""\n\n'
        + "LOGO_ASCII = "
        + repr(art)
        + "\n\nRGB_ROWS = (\n"
        + "".join("    " + repr(row) + ",\n" for row in colors)
        + ")\n"
    )
    output = root / "src/emsuite/_logo.py"
    output.write_text(text)
    print(f"Generated {output.name}: {width} columns, {len(lines)} rows, original RGB colors")


if __name__ == "__main__":
    main()
