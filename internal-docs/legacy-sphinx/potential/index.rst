The **potential** channel maps APBS electrostatics onto VDW surface points and writes
a heterogeneous ``.surf`` file.

CLI
---

.. code-block:: bash

   emsuite -p potential.in

User choice
-----------

- ``quantity = 'potential'`` — interpolate the APBS potential grid onto surface coordinates
- ``quantity = 'charge'`` — Gauss-law charges from that potential and the APBS dielectric maps
  (``ρ = −ε₀ ∇·(ε ∇φ)``)

Engine
------

- ``method = 'apbs'`` — Poisson–Boltzmann (default). Writes potential and ``dielx/y/z`` maps.
- ``method = 'coulomb'`` — vacuum ``q/r`` fallback. Potential only; cannot be combined with ``quantity='charge'``.
- Planned: ``esp`` / ``mep`` via PySCF.

See :doc:`../ROADMAP` and ``examples/templates/potential.in``.
