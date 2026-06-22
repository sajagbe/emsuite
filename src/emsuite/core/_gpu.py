try:
    import cupy as cp

    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False
