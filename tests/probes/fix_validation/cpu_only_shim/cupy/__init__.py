# Put this directory first on PYTHONPATH to make `import cupy` fail, so emsuite runs
# exactly as a CPU-only install (`pip install emsuite`, no [gpu] extra) would.
raise ImportError("cupy disabled by tests/probes/fix_validation/cpu_only_shim (CPU validation run)")
