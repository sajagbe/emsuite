Files
=====

Specification of input and output file formats.

.. contents:: On This Page
   :local:
   :depth: 1

ETM File (Surface Definition)
-----------------------------

The ``.etm`` (Electrostatic Tuning Map) file defines probe positions 
around the molecule.

Homogeneous Format
^^^^^^^^^^^^^^^^^^

For uniform charge at all points (3 columns):

.. code-block:: text

   x          y          z         
   0.969663   1.451681   -0.346153 
   0.004572   -0.011758  1.637124  
   -0.393951  0.847387   1.315983

- **Line 1**: Column headers (required)
- **Lines 2+**: X, Y, Z coordinates in Ångströms

The charge at each point is set by ``surface_charge`` in the input file.

Heterogenous Format
^^^^^^^^^^^^^^^^^^^

For different charges at each point (4 columns):

.. code-block:: text

   x          y          z          q
   0.969663   1.451681   -0.346153  0.5
   0.004572   -0.011758  1.637124   -0.3
   -0.393951  0.847387   1.315983   1.0

- **Column 4 (q)**: Point charge in elementary charge units (*e*)

Auto-Generation
^^^^^^^^^^^^^^^

If the ETM file doesn't exist and ``surface_type = 'homogenous'``, 
emsuite auto-generates it from the molecule's VDW surface using 
the ``scale`` and ``density`` parameters.

MOL2 File (Visualization)
-------------------------

Two MOL2 files are created for each property:

- ``{molecule}_{property}.mol2`` — Raw effect values
- ``{molecule}_{property}_normalized.mol2`` — Scaled to [-1, 1]

Structure
^^^^^^^^^

.. code-block:: text

   @<TRIPOS>MOLECULE
   homo | baseline=-13.547519
      30 0 0 0
   SMALL
   GASTEIGER
   @<TRIPOS>ATOM
       1 H      0.9697   1.4517  -0.3462 H1   1 HOMO      -0.842870
       2 H      0.0046  -0.0118   1.6371 H1   1 HOMO      -0.898435

**Header section:**

- Line 2: ``{property} | baseline={value}`` — Property name and baseline value

**Atom section:**

- Column 1: Atom index
- Columns 2-4: X, Y, Z coordinates
- Column 5: Atom type
- Column 6: Residue number
- Column 7: Property name
- Column 8: Effect value (stored as ``partial_charge``)

Visualization in PyMOL
^^^^^^^^^^^^^^^^^^^^^^

**For normalized files:**

.. code-block:: python

   alter all, b=partial_charge
   spectrum b, red_white_blue, minimum=-1, maximum=1

**For raw files:**

.. code-block:: python

   alter all, b=partial_charge
   select high, partial_charge > 0.25
   color blue, high
   select low, partial_charge < -0.25
   color red, low
   select neutral, partial_charge > -0.25 and partial_charge < 0.25
   color white, neutral

CSV File (Data Summary)
-----------------------

``{molecule}_tuning_summary.csv`` contains all calculated data in 
tabular format.

Structure
^^^^^^^^^

.. code-block:: text

   point_index,x,y,z,homo_effect,homo_effect_normalized,homo_baseline
   0,0.969663,1.451681,-0.346153,-0.8429,-0.0052,-13.5475
   1,0.004572,-0.011758,1.637124,-0.8984,-0.2875,-13.5475

Columns
^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Description
   * - ``point_index``
     - Surface point index (0-based)
   * - ``x``, ``y``, ``z``
     - Coordinates in Ångströms
   * - ``{property}_effect``
     - Change in property (with charge − without)
   * - ``{property}_effect_normalized``
     - Effect scaled to [-1, 1] range
   * - ``{property}_baseline``
     - Property value without surface charge

XYZ File (Molecule Geometry)
----------------------------

Standard XYZ format for molecular coordinates:

.. code-block:: text

   3

   O  0.000000  0.000000  0.117176
   H  0.000000  0.755453 -0.468706
   H  0.000000 -0.755453 -0.468706

- **Line 1**: Number of atoms
- **Line 2**: Comment (can be blank)
- **Lines 3+**: Element symbol and X, Y, Z coordinates in Ångströms
