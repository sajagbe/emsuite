The **potential** channel computes electrostatic potential at VDW surface points and writes
heterogeneous ``.surf`` files for downstream tuning or coupled workflows.

CLI
---

.. code-block:: bash

   emsuite -p potential.in

Methods
-------

- ``method = 'coulomb'`` — Gasteiger partial charges (default, fast, CI-friendly)
- ``method = 'apbs'`` — Poisson-Boltzmann via ``apbs-binary`` (falls back to Coulomb on failure)

See :doc:`../ROADMAP` for architecture and ``examples/templates/potential.in``.
