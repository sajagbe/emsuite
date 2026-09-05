Combined
========

The **coupled** channel runs electrostatic potential mapping, then feeds the
resulting heterogeneous ``.surf`` file into tuning.

CLI
---

.. code-block:: bash

   emsuite -c coupled.in

This is **not** the same as ``calc_type='combined'`` inside ``tuning.in``,
which applies all homogeneous probe charges at once.

See :doc:`../ROADMAP` and ``examples/templates/coupled.in``.
