Output
======

Results are organized into a timestamped folder for each calculation.

Directory Layout
----------------

.. code-block:: text

   results_{molecule}_{timestamp}/
   ├── {molecule}_{property}.mol2
   ├── {molecule}_{property}_normalized.mol2
   ├── {molecule}_tuning_summary.csv
   ├── README.txt
   └── logs/
       ├── .resume_metadata.json
       └── point_*.log

Output Files
------------

MOL2 Files
^^^^^^^^^^

Two files per property:

``{molecule}_{property}.mol2``
   Raw effect values for visualization.

``{molecule}_{property}_normalized.mol2``
   Values scaled to [-1, 1] for consistent color mapping.

For excited states, files are numbered by state:

- ``molecule_s1_exe.mol2`` (singlet state 1)
- ``molecule_s2_exe.mol2`` (singlet state 2)
- ``molecule_t1_exe.mol2`` (triplet state 1)

CSV Summary
^^^^^^^^^^^

``{molecule}_tuning_summary.csv``
   Complete data table with coordinates, effects, normalized values, 
   and baselines for all properties.

README
^^^^^^

``README.txt``
   Calculation metadata including molecule name, timestamp, 
   properties calculated, and file descriptions.

Logs Directory
--------------

The ``logs/`` folder contains:

``.resume_metadata.json``
   State information for resuming interrupted calculations.
   If a calculation is interrupted, emsuite can continue from 
   the last completed point.

``point_*.log``
   Individual log files for each surface point calculation.
   Useful for debugging failed points.

Resuming Calculations
---------------------

If a calculation is interrupted, simply run the same command again:

.. code-block:: bash

   emsuite -t tuning.in

emsuite detects the ``.resume_metadata.json`` file and continues 
from where it left off.

Example Output
--------------

For a water molecule with ``properties = ['homo', 'lumo']``:

.. code-block:: text

   results_water_2026-01-19_14-30-00/
   ├── water_homo.mol2
   ├── water_homo_normalized.mol2
   ├── water_lumo.mol2
   ├── water_lumo_normalized.mol2
   ├── water_tuning_summary.csv
   ├── README.txt
   └── logs/
       ├── .resume_metadata.json
       ├── point_000.log
       ├── point_001.log
       └── ...
