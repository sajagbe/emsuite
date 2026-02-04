Installation
============


.. code-block:: bash

   pip install emsuite


or:

.. code-block:: bash

   pip install emsuite[gpu]


Dependencies
------------

CPU
^^^

CPU installation includes:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Package
     - Purpose
   * - pyscf
     - Quantum chemistry calculations
   * - rdkit
     - Molecular structure handling and SMILES parsing
   * - geometric
     - Geometry optimization
   * - vdw-surfgen
     - Van der Waals surface generation
   * - ray
     - Parallel processing
   * - numpy
     - Numerical operations
   * - requests
     - HTTP requests for external resources

GPU
^^^

Installs everything from CPU, plus:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Package
     - Version
     - Purpose
   * - gpu4pyscf-cuda12x
     - 1.4.3
     - GPU-accelerated quantum chemistry
   * - gpu4pyscf-libxc-cuda12x
     - 0.5
     - GPU-accelerated exchange-correlation functionals
   * - cupy-cuda12x
     - 13.6.0
     - GPU array operations
   * - cutensor-cu12
     - 2.0.2
     - CUDA tensor operations

.. warning::

    Avoid changing gpu4pyscf, cupy, and cutensor versions as they are intentionally selected to maintain mutual compatibility.

.. note::
   emsuite automatically detects your hardware. 
   If GPU libraries are installed but no compatible GPU is found, it falls back to CPU mode.

Requirements
-------------------

**General:**

- Python 3.11 or higher
- Linux, macOS, or Windows

**GPU:**

- NVIDIA GPU with CUDA 12.x support
- NVIDIA drivers (version 525 or higher recommended)
- ~4 GB free disk space for CUDA libraries


Verification
-------------------

To check that emsuite is installed correctly:

.. code-block:: bash

   emsuite --help

You should see:

.. code-block:: text

   usage: emsuite [-h] (-t INPUT_FILE)

   EMSuite - Electrostatic Map Suite

   options:
     -h, --help            show this help message and exit
     -t INPUT_FILE, --tuning INPUT_FILE
                           Run electrostatic tuning calculation

Virtualization (Recommended)
---------------------------------

We recommend installing emsuite in a virtual environment:

.. code-block:: bash

   # Create virtual environment
   python -m venv emsuite-env

   # Activate it
   source emsuite-env/bin/activate   # Linux/macOS
   emsuite-env\Scripts\activate      # Windows

   # Install
   pip install emsuite

Troubleshooting
---------------

**ModuleNotFoundError: No module named 'pyscf'**

Your installation may be incomplete. Try reinstalling:

.. code-block:: bash

   pip uninstall emsuite
   pip install emsuite

**GPU not detected**

Verify CUDA is available:

.. code-block:: bash

   python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"

If this returns 0 or errors, check your NVIDIA drivers and CUDA installation.

**Permission errors on Linux**

Use ``--user`` flag or a virtual environment:

.. code-block:: bash

   pip install --user emsuite

