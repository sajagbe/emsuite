Inputs
======

The tuning module reads parameters from a ``.in`` file (typically ``tuning.in``).
The file uses Python-style syntax with ``key = value`` pairs.
For example:

.. code-block:: python

   # Input molecule as SMILES
   input_type = 'SMILES'
   input_data = 'O'
   properties = ['homo', 'lumo']


.. tip:: 
  Comment out any line in the input file with the # sign.


The input file is split into 5 sections, 3 of which are optional, viz:

1. **Molecule** - Defining the input molecule (*required*).
2. **Properties** - Defining properties to compute (*required*).
3. **Methods** - Defining Quantum Mechanical methods (*optional*).
4. **Surface** - Defining surface and probe charge configurations  (*optional*).
5. **Parallelism** - Defining parallel execution (*optional*).

Molecule
--------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``input_type``
     - str
     - *required*
     - Input format: ``'SMILES'`` or ``'XYZ'``
   * - ``input_data``
     - str
     - *required*
     - SMILES string or path to XYZ file (ideally should be in same folder as input file for ease). 
   * - ``charge``
     - int
     - ``0``
     - Molecular charge
   * - ``spin``
     - int
     - ``0``
     - Spin multiplicity (based on PySCF API, where, 0=singlet, 1=doublet, etc).


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
     - *required*
     - Properties to calculate (see :doc:`properties`)
   * - ``state_of_interest``
     - int
     - ``1``
     - Number of excited states (for ``'exe'`` and ``'osc'`` calculations).
   * - ``triplet``
     - bool
     - ``False``
     - Calculate triplet states instead of singlets.

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
     - Calculation method: ``'dft'`` or ``'hf'``
   * - ``functional``
     - str
     - ``'b3lyp'``
     - DFT functional (used for ``method='dft'`` only), functional name or code can be used as string. Extensive list available on `GitHub <https://github.com/sajagbe/emsuite/blob/main/method-info/functionals.csv>`_.
   * - ``basis_set``
     - str
     - ``'6-31G*'``
     - Basis set, extensive list available on `GitHub <https://github.com/sajagbe/emsuite/blob/main/method-info/basis-sets>`_.
   * - ``solvent``
     - str or None
     - ``None``
     - Solvent name for implicit solvation, or ``None`` for gas phase. See list of solvents on `GitHub <https://github.com/sajagbe/emsuite/blob/main/method-info/solvents.csv>`_.


Surface
-------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``surface_type``
     - str
     - ``'homogenous'``
     - ``'homogenous'`` (uniform charge at each surface point), or 

       ``'heterogenous'`` (unique charge at each surface point).
   * - ``surface_file``
     - str
     - ``'surface.etm'``
     - Path to surface file (auto-generated if file unavailable). 
        
       For ``'heterogenous'`` ``surface_type``, this file must exist and can be created manually using `vdw-surfgen <https://pypi.org/project/vdw-surfgen/>`_ in the format described in :doc:`files`.
   * - ``surface_charge``
     - float
     - ``1.0``
     - Point charge magnitude in *e* (used for ``'homogenous'`` ``surface_type`` only).
   * - ``scale``
     - float
     - ``1.0``
     - VdW surface scaling factor (used for ``'homogenous'`` ``surface_type`` only).
   * - ``density``
     - float
     - ``1.0``
     - Surface point density (used for ``'homogenous'`` ``surface_type`` only).
   * - ``calc_type``
     - str
     - ``'separate'``
     - ``'separate'`` (one charge at a time in separate calculation) or ``'combined'`` (all charges together in a single calculation)

Parallelism
-----------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``parallel``
     - bool
     - ``True``
     - Enable parallel processing
   * - ``num_procs``
     - int
     - ``4``
     - Number of GPU/CPU cores to engage in parallel.

Examples
--------

**Minimal example:**

.. code-block:: python

   input_type = 'SMILES'
   input_data = 'O'
   properties = ['homo']

**Full example:**

.. code-block:: python

   # Molecule
   input_type = 'SMILES'
   input_data = 'O'

   # Properties
   properties = ['homo', 'lumo', 'gap']
   state_of_interest = 3
   triplet = False

   # Methods
   method = 'dft'
   functional = 'b3lyp'
   basis_set = '6-31G*'
   charge = 0
   spin = 0
   solvent = 'ethanol'

   # Surface
   surface_type = 'homogenous'
   surface_charge = 1.0
   surface_file = 'molecule.etm'
   calc_type = 'separate'
   scale = 1.0
   density = 1.0

   # Parallelism
   parallel = True
   num_procs = 8
