import os

from ._gpu import GPU_AVAILABLE, cp


def check_gpu_info():
    """
    This function uses CuPy to detect CUDA-capable GPUs on the system.
    """
    if not GPU_AVAILABLE:
        print("\nCuPy not installed - CPU mode only.")
        print("For GPU acceleration: pip install emsuite[gpu]\n")
        return 0

    try:
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count < 1:
            print("\nNo GPUs found.\nSwitching to CPU mode.\n")
            return 0
        else:
            print(f"\n{device_count} GPU(s) detected.\n")
            return device_count
    except Exception as e:
        print(f"\nGPU not available: {e}")
        print("Switching to CPU mode.\n")
        return 0


def check_cpu_info():
    """
    Get the number of available CPU cores on the system.

    Returns:
        int: Number of CPU cores available, defaults to 1 if unable to determine

    Note:
        Uses os.cpu_count() to detect CPU cores and handles exceptions.
    """
    try:
        cpu_cores = os.cpu_count()
        # print(f"Number of CPU cores available: {cpu_cores}")
        return cpu_cores
    except Exception as e:
        print(f"Could not determine CPU cores: {e}")
        return 1  # Default to 1 if unable to determine


##############################################
#              Print Messages                #
##############################################


def print_startup_message():
    """
    Print the startup banner message for the Electrostatic Map Suite.

    """
    print("\n")
    print("=" * 60)
    print("                   Electrostatic Map Suite")
    print("                    By Stephen O. Ajagbe")
    print("=" * 60)


##############################################
