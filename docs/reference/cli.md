---
icon: lucide/terminal
---

# Command-Line Interface

```text
usage: emsuite [-h] (-t INPUT_FILE | -s INPUT_FILE | -p INPUT_FILE | -c INPUT_FILE)
```

Exactly one channel is required.

| Option | Long form | Action |
| --- | --- | --- |
| `-s` | `--surface` | Generate a van der Waals surface |
| `-p` | `--potential` | Compute APBS values on a surface |
| `-t` | `--tuning` | Compute electrostatic tuning maps |
| `-c` | `--coupled` | Run Potential followed by Tuning |
| `-h` | `--help` | Show command help |

Examples:

```bash
emsuite -s surface.in
emsuite -p potential.in
emsuite -t tuning.in
emsuite -c coupled.in
```

Input paths are checked before the channel is loaded. Relative paths inside an
input file resolve from the process working directory, so launch EMSuite from a
directory that makes those paths unambiguous.
