import json
import os
import urllib.error
import urllib.request

from ._gpu import CUPY_AVAILABLE, cp

OFFICE_API = "https://officeapi.akashrajpurohit.com"


def check_gpu_info():
    """
    This function uses CuPy to detect CUDA-capable GPUs on the system.
    """
    if not CUPY_AVAILABLE:
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


def print_office_quote() -> None:
    """Fetch and print a random Office quote after a successful job.

    Network failures are reported but do not raise — a finished calculation
    should not fail because of the Easter egg.
    """
    print("\nFetching inspirational quote...\n")

    url = f"{OFFICE_API}/quote/random"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "emsuite/1.6 (+https://github.com/sajagbe/emsuite)"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        quote = data.get("quote", "").strip()
        character = data.get("character", "").strip()
        if quote and character:
            print(f"\n  {quote} \n                     - {character}\n")
        else:
            print("(Office quote API returned an unexpected payload.)\n")
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        json.JSONDecodeError,
        OSError,
    ) as exc:
        print(f"(Could not fetch Office quote: {exc})\n")


##############################################
