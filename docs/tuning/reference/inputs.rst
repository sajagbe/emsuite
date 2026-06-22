Inputs
======

The tuning module reads parameters from a ``tuning.in`` file using ``key = value`` pairs
(parsed safely with ``ast.literal_eval``). Surface generation is a **separate** step via
``emsuite -s surface.in``; tuning expects an existing XYZ and ``.surf`` file.

Example:

.. code-block:: python

   molecule = 'CCO_opt.xyz'
   surface_file = 'CCO.surf'
   properties = ['exe']
   calc_type = 'separate'

See also: :doc:`../../ROADMAP` and the `README <https://github.com/sajagbe/emsuite/blob/main/README.md>`_.

Molecule
--------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``molecule`` or ``xyz_file``
     - str
     - *required*
     - Path to XYZ geometry file
   * - ``surface_file``
     - str
     - *required*
     - Path to ``.surf`` file from surface generation
   * - ``charge``
     - int
     - ``0``
     - Molecular charge
   * - ``spin``
     - int
     - ``0``
     - Spin (PySCF 2S notation: 0=singlet, 1=doublet, etc.)

Properties
----------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``properties``
     - list
     - ``['all']``
     - Properties to calculate (see :doc:`properties`)
   * - ``state_of_interest``
     - int
     - ``2``
     - Excited state index for ``exe`` / ``osc``
   * - ``triplet``
     - bool
     - ``False``
     - Calculate triplet states instead of singlets

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``method``
     - str
     - ``'dft'``
     - ``'dft'`` or ``'hf'``
   * - ``functional``
     - str
     - ``'b3lyp'``
     - DFT functional
   * - ``basis_set``
     - str
     - ``'6-31G*'``
     - Basis set
   * - ``solvent``
     - str or None
     - ``None``
     - Implicit solvation solvent name, or gas phase

Surface / execution
-------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``calc_type``
     - str
     - ``'separate'``
     - ``'separate'`` (one probe at a time) or ``'combined'`` (all charges at once)
   * - ``parallel``
     - bool
     - ``True``
     - Enable Ray parallel execution
   * - ``num_procs``
     - int or None
     - ``None``
     - Worker count (auto-detect if ``None``)
