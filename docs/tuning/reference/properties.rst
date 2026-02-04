Properties
==========

Available molecular properties for tuning calculations.

Property Codes
--------------

.. list-table::
   :header-rows: 1
   :widths: 10 40 15 35

   * - Code
     - Description
     - Units
     - Dependencies
   * - ``'gse'``
     - Ground state energy
     - kcal/mol
     - —
   * - ``'homo'``
     - HOMO energy 
     - eV
     - —
   * - ``'lumo'``
     - LUMO energy
     - eV
     - —
   * - ``'gap'``
     - HOMO-LUMO gap
     - eV
     - homo, lumo
   * - ``'dm'``
     - Dipole moment magnitude
     - Debye
     - —
   * - ``'ie'``
     - Ionization energy
     - kcal/mol
     - cation calculation
   * - ``'ea'``
     - Electron affinity
     - kcal/mol
     - anion calculation
   * - ``'cp'``
     - Chemical potential
     - kcal/mol
     - ie, ea
   * - ``'eng'``
     - Electronegativity
     - eV
     - cp
   * - ``'hard'``
     - Chemical hardness
     - eV
     - ie, ea
   * - ``'efl'``
     - Electrophilicity index
     - eV
     - cp, hard
   * - ``'nfl'``
     - Nucleophilicity index
     - eV
     - efl
   * - ``'exe'``
     - Excitation energies
     - eV
     - TD calculation
   * - ``'osc'``
     - Oscillator strengths
     - dimensionless
     - TD calculation

Using Properties
----------------

Specify properties as a list in your input file:

.. code-block:: python

   # Single property
   properties = ['homo']

   # Multiple properties
   properties = ['homo', 'lumo', 'gap', 'ie', 'ea']

   # All properties
   properties = ['all']

Property Dependencies
---------------------

Some properties require additional calculations:

**Derived properties** (calculated from other properties):

- ``gap`` requires ``homo`` and ``lumo``
- ``cp`` requires ``ie`` and ``ea``
- ``eng`` requires ``cp``
- ``hard`` requires ``ie`` and ``ea``
- ``efl`` requires ``cp`` and ``hard``
- ``nfl`` requires ``efl``

**Ion calculations** (run separate SCF for cation/anion):

- ``ie`` requires cation calculation
- ``ea`` requires anion calculation

**TD calculations** (time-dependent for excited states):

- ``exe`` requires TD-DFT/TD-HF
- ``osc`` requires TD-DFT/TD-HF

Dependencies are resolved automatically. If you request ``gap``, 
emsuite will calculate ``homo`` and ``lumo`` as well.

Excited State Properties
------------------------

For ``exe`` and ``osc``, use these additional parameters:

.. code-block:: python

   properties = ['exe', 'osc']
   state_of_interest = 3    # Calculate 3 excited states
   triplet = False          # Singlet states (default)

This generates separate output files for each state:

- ``molecule_s1_exe.mol2`` (first singlet)
- ``molecule_s2_exe.mol2`` (second singlet)
- ``molecule_s3_exe.mol2`` (third singlet)

For triplet states:

.. code-block:: python

   triplet = True

Output files will be named ``molecule_t1_exe.mol2``, etc.
