CLI
===

Usage
-----

.. code-block:: bash

   emsuite -t <input_file>
   emsuite --tuning <input_file>

Arguments
---------

``-t``, ``--tuning``
   Run a tuning calculation with the specified input file.

``-h``, ``--help``
   Show help message and exit.

Examples
--------

**Basic usage:**

.. code-block:: bash

   emsuite -t tuning.in

**With full path:**

.. code-block:: bash

   emsuite -t /path/to/my_calculation.in

**Show help:**

.. code-block:: bash

   emsuite --help

Output
------

.. code-block:: text

   ╔══════════════════════════════════════════════════════════════════╗
   ║                           EMSuite                                ║
   ║           Electrostatic Mapping Suite v1.0.4                     ║
   ╚══════════════════════════════════════════════════════════════════╝

   Reading input: tuning.in
   Generating VDW surface...
   Running calculations: 100%|████████████████████| 47/47

   Results saved to: results_water_2026-01-19_10-30-00/

Exit Codes
----------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Code
     - Meaning
   * - ``0``
     - Success
   * - ``1``
     - Input file not found or invalid
   * - ``2``
     - Calculation error
