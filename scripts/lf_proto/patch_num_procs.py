#!/usr/bin/env python3
"""Patch num_procs in tuning.in or coupled.in from SLURM_GPUS_ON_NODE."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

NUM_PROCS_RE = re.compile(
    r"^(\s*num_procs\s*=\s*)(?:\d+|None)(\s*(?:#.*)?)$",
    re.MULTILINE,
)


def patch_num_procs(path: Path, num_procs: int) -> None:
    text = path.read_text()
    if NUM_PROCS_RE.search(text):
        text = NUM_PROCS_RE.sub(rf"\g<1>{num_procs}\g<2>", text)
    else:
        if not text.endswith("\n"):
            text += "\n"
        text += f"num_procs = {num_procs}\n"
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=Path, help="tuning.in or coupled.in to patch")
    parser.add_argument(
        "num_procs",
        nargs="?",
        type=int,
        default=None,
        help="Explicit process count (overrides env/default)",
    )
    parser.add_argument(
        "--from-env",
        default="SLURM_GPUS_ON_NODE",
        help="Environment variable holding process count (default: SLURM_GPUS_ON_NODE)",
    )
    parser.add_argument(
        "--default",
        type=int,
        default=4,
        help="Fallback when env var is unset (default: 4)",
    )
    args = parser.parse_args()

    if args.num_procs is not None:
        num_procs = args.num_procs
        source = f"arg={num_procs}"
    else:
        raw = os.environ.get(args.from_env)
        num_procs = int(raw) if raw else args.default
        source = f"{args.from_env}={raw!r}"
    patch_num_procs(args.input_file, num_procs)
    print(f"Patched {args.input_file} -> num_procs = {num_procs} ({source})")


if __name__ == "__main__":
    main()
