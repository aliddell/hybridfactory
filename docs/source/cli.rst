Command-line interface
----------------------

Hybrid Factory is primarily a command-line tool.
Provided you have a :ref:`parameter file <parameter-file>`, you can invoke it like so:

.. code:: shell

    (hybridfactory) $ hybridfactory generate /path/to/params.py

Right now, ``generate`` is the only command available, allowing you to
generate hybrid data from a pre-existing raw data set and output from a
spike-sorting tool, e.g.,
`KiloSort <https://github.com/cortex-lab/KiloSort>`__ or
`JRCLUST <https://github.com/JaneliaSciComp/JRCLUST>`__.
This is probably what you want to do.

The new hybrid dataset will be output in a directory of your choosing (see
``output_directory`` in the
:ref:`parameter file <parameter-file>`).
