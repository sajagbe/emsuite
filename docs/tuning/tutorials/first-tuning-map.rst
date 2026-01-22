Your First Electrostatic Tuning Map
===================================

In this tutorial, you will calculate the effects of placing a +1\ *e* probe charge across 
the VdW surface of water on the HOMO energy (in eV),  — and see the results in under 5 minutes.

Preface: Creating a Molecule File
---------------------------------

Create a file called ``water.xyz`` with these contents:

.. code-block:: text

   3

   O  0.000000  0.000000  0.117176
   H  0.000000  0.755453 -0.468706
   H  0.000000 -0.755453 -0.468706

This is a water molecule in XYZ format. The first line is the atom count, 
the second line is blank (or a comment), and the remaining lines are atom 
coordinates in Ångströms.



Creating an Input File
-----------------------

Create a file called ``tuning.in`` - or another name of your choice - in the same folder:

.. code-block:: python

   # Molecule
   input_type = 'XYZ'
   input_data = 'water.xyz'

   # Calculation settings
   method = 'dft'
   functional = 'b3lyp'
   charge = 0
   spin = 0

   # What to calculate
   properties = ['homo']
   surface_charge = 1.0

That's it. This tells emsuite to:

- Read the water molecule from ``water.xyz``
- Use DFT with the B3LYP functional
- Calculate how the HOMO energy changes when a +1 charge is placed around the molecule

.. tip::

   You can also bypass this and use SMILES notation in the input file instead of XYZ files. 
   For water, that would be ``input_type = 'SMILES'`` and ``input_data = 'O``.

Running the Calculation
-----------------------

Open a terminal in the folder containing both files and run:

.. code-block:: bash

   emsuite -t tuning.in

You'll see output:

.. code-block:: text

   ╔══════════════════════════════════════════════════════════════════╗
   ║                           EMSuite                                ║
   ║           Electrostatic Mapping Suite v1.0.4                     ║
   ╚══════════════════════════════════════════════════════════════════╝

   Reading input: tuning.in
   Generating VDW surface...
   *Details of calculation*
   Results saved to: results_water_2026-01-19_10-30-00/

This takes about 1–2 minutes on a typical laptop.

Viewing Your Results
--------------------

Look in the results folder. You'll find:

``water_homo.mol2``
   Raw tuning values. Use for custom analysis.

``water_homo_normalized.mol2``
   Values scaled to [-1, 1] range. Best for visualization.

``water_tuning_summary.csv``
   A spreadsheet with all the data. Each row contains:
   
   - X, Y, Z coordinates of a surface point
   - The change in HOMO energy (in eV) when a +1 charge is placed at that point

Visualizing in PyMOL
-----------------------

Open your results:

.. code-block:: bash

   pymol water_homo_normalized.mol2

**For normalized files** (recommended):

Run these commands in PyMOL's command line:

.. code-block:: python

   alter all, b=partial_charge
   spectrum b, red_white_blue
   spectrum b, red_white_blue, minimum=-1, maximum=1

This creates a smooth gradient:

- **Red**: Negative values (property decreases)
- **White**: Near zero (no change)  
- **Blue**: Positive values (property increases)

**For non-normalized files:**

Use manual thresholds to highlight significant changes, for example:

.. code-block:: python

   alter all, b=partial_charge
   select high_charge, partial_charge > 0.25
   color blue, high_charge
   select low_charge, partial_charge < -0.25
   color red, low_charge
   select neutral_charge, partial_charge > -0.25 and partial_charge < 0.25
   color white, neutral_charge

Adjust the threshold values (0.25) based on your data range.

.. tip::

   Check your actual data range in the CSV file to set appropriate 
   thresholds for non-normalized visualization.

What Just Happened?
-------------------

emsuite placed a +1 point charge at ~47 positions on a surface around 
the water molecule. At each position, it ran a quantum chemistry 
calculation to measure how the HOMO energy changed.

The result is an **electrostatic tuning map** — showing which regions 
around a molecule are sensitive to external charges and the relative effects of these regions.

Next Steps
----------

Try modifying your input file:

**Calculate multiple properties:**

.. code-block:: python

   properties = ['homo', 'lumo', 'gap']

**Use a negative probe charge:**

.. code-block:: python

   surface_charge = -1.0

**Calculate excited state tuning:**

.. code-block:: python

   properties = ['exe']
   state_of_interest = 1

**Use SMILES instead of XYZ:**

.. code-block:: python

   input_type = 'SMILES'
   input_data = 'O'

See the Tuning Reference :doc:`../reference/inputs` page for all available options.
