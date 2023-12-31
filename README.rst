Hybrid ground-truth generation for spike-sorting
================================================

Installation
------------

.. _install:

For full documentation, go
`here <https://vidriotech.gitlab.io/hybridfactory>`__.

Getting started
~~~~~~~~~~~~~~~

The best way to get started is to `install Anaconda or Miniconda`_.
Once you've done that, fire up your favorite terminal emulator (PowerShell or
CMD on Windows, but we recommend CMD; iTerm2 or Terminal on Mac; lots of
choices if you're on Linux, but you knew that) and navigate to the base
directory of the repository (it should contain ``environment.yml``).

On UNIX variants, type:

.. code:: bash

    $ conda env create -n hybridfactory -f environment.yml
    Solving environment: done
    Downloading and Extracting Packages
    ...
    Preparing transaction: done
    Verifying transaction: done
    Executing transaction: done
    #
    # To activate this environment, use
    #
    #     $ conda activate hybridfactory
    #
    # To deactivate an active environment, use
    #
    #     $ conda deactivate

    $ conda activate hybridfactory

On Windows:

.. code:: shell

    $ conda env create -n hybridfactory -f environment.yml
    Solving environment: done
    Downloading and Extracting Packages
    ...
    Preparing transaction: done
    Verifying transaction: done
    Executing transaction: done
    #
    # To activate this environment, use:
    # > activate hybridfactory
    #
    # To deactivate an active environment, use:
    # > deactivate
    #
    # * for power-users using bash, you must source
    #

    $ activate hybridfactory

and you should be good to go.
**Remember that**
``[conda] activate hybridfactory`` **every time you open up a new shell!**


Usage
-----

Hybrid Factory is primarily a command-line tool.
Provided you have a `parameter file`_, you can invoke it like so:

.. code:: shell

    (hybridfactory) $ hybridfactory generate /path/to/params.py

Right now, ``generate`` is the only command available, allowing you to
generate hybrid data from a pre-existing raw data set and output from a
spike-sorting tool, e.g., `KiloSort`_ or `JRCLUST`_.
This is probably what you want to do.

The new hybrid dataset will be output in a directory of your choosing (see
``output_directory`` in the `parameter file`_).

After your hybrid data has been generated, we have some
`validation tools`_ which you can use to look at your
hybrid output, but this is not as convenient as a command-line tool (yet).

A word about bugs
~~~~~~~~~~~~~~~~~

This software is under active development. Although we strive for
accuracy and consistency, there's a good chance you'll run into some
bugs. If you run into an exception which crashes the program, you should
see a helpful message with my email address and a traceback. If you find
something a little more subtle, please post an issue on the `issue page`_.

Parameter file
--------------

Rather than pass a bunch of flags and arguments to ``hybridfactory``, we
have collected all the parameters in a parameter file, ``params.py``.
We briefly explain each option below.
See the `example parameter file`_ for usage.

Required parameters
~~~~~~~~~~~~~~~~~~~

-  ``data_directory``: Directory containing output from your spike
   sorter, e.g., ``rez.mat`` or ``*.npy`` for KiloSort; or ``*_jrc.mat``
   and ``*_spk(raw|wav|fet).jrc`` for JRCLUST.
   This does *not* have to contain your raw source file.
-  ``raw_source_file``: Path to file containing raw source data
   (currently only
   SpikeGLX_-formatted data is supported).
   This path can also be a glob_ if you have multiple data files.
-  ``data_type``: Type of raw data, as a `NumPy data type`_.
   (I have only seen ``int16``.)
-  ``sample_rate``: Sample rate of the source data, in Hz.
-  ``ground_truth_units``: Cluster labels (1-based indexing) of
   ground-truth units from your spike sorter's output.
-  ``start_time``: Start time (0-based) of recording in data file (in
   sample units). Nonnegative integer if ``raw_source_file`` is a single
   file, iterable of nonnegative integers if you have a globbed
   ``raw_source_file``.
   **Note:** We perform a `natural sorting`_ of globbed files, which should
   yield a strictly increasing sequence of start times.
   Manually-supplied start times should reflect this, i.e., should be sorted.
   If you have SpikeGL meta files, you can use
   ``hybridfactory.io.spikegl.get_start_times`` to get these
   automatically.
-  ``probe``: A model of the probe used to collect and sort the data.
   See the `section <#probe-configuration>`__ on probe configuration for
   details.

Optional parameters
~~~~~~~~~~~~~~~~~~~

-  ``session_name``: String giving an identifying name to your hybrid
   run. Default is an MD5 hash computed from the current timestamp.
-  ``random_seed``: Nonnegative integer in the range
   :math:``[0, 2^{31})``. Because this algorithm is randomized, setting
   a random seed allows for reproducible output. The default is itself
   randomly generated, but will be output in a
   ``hfparams_[session_name].py`` on successful completion.
-  ``output_directory``: Path to directory where you want to output the
   hybrid data. (This includes raw data files and annotations.) Defaults
   to "``data_directory``/hybrid\_output".
-  ``output_type``: Type of output from your spike sorter. One of "phy"
   (for ``*.npy``), "kilosort" (for ``rez.mat``), or "jrc" (for
   ``*_jrc.mat`` and ``*_spk(raw|wav|fet).jrc``). ``hybridfactory`` will
   try to infer it from files in ``data_directory`` if not specified.
-  ``num_singular_values``: Number of singular values to use in the
   construction of artificial events. Default is 6.
-  ``channel_shift``: Number of channels to shift artificial events up
   or down from their source. Default depends on the probe used.
-  ``synthetic_rate``: Firing rate, in Hz, for hybrid units. This should
   be either an empty list (if you want to use the implicit firing rate
   of your ground-truth units) or an iterable of artificial rates. In
   the latter case, you must specify a firing rate for each ground-truth
   unit. Default is the implicit firing rate of each unit.
-  ``time_jitter``: Scale factor for (normally-distributed) random time
   shift, in sample units. Default is 100.
-  ``amplitude_scale_min``: Minimum factor for (uniformly-distributed)
   random amplitude scaling, in percentage units. Default is 1.
-  ``amplitude_scale_max``: Maximum factor for (uniformly-distributed)
   random amplitude scaling, in percentage units. Default is 1.
-  ``samples_before``: Number of samples to take before an event
   timestep for artificial event construction. Default is 40.
-  ``samples_after``: Number of samples to take after an event timestep
   for artificial event construction. Default is 40.
-  ``copy``: Whether or not to copy the source file (the original raw data) to
   the target (the new raw data file containing the hybrid units).
   You usually want to do this, but if the file is large and you know where
   your data has been perturbed, you could use
   |HybridDataSet.reset|_ instead. Default is False.

Probe configuration
~~~~~~~~~~~~~~~~~~~

-  ``probe_type``: Probe layout. This is pretty open-ended so it is up
   to you to construct. If you have a Neuropixels Phase 3A probe with
   the standard reference channels, you have it easy. Just put
   ``neuropixels3a()`` for this value. Otherwise, you'll need to
   construct the following NumPy arrays to describe your probe:
-  ``channel_map``: a 1-d array of ``n`` ints describing which row in
   the data to look for which channel (0-based).
-  ``connected``: a 1-d array of ``n`` bools, with entry ``k`` being
   ``True`` if and only if channel ``k`` was used in the sorting.
-  ``channel_positions``: an :math:``n \times 2`` array of floats, with
   row ``k`` holding the x and y coordinates of channel
   ``channel_map[k]``.
-  ``name`` (optional): a string giving the model name of your probe.
   This is just decorative for now.

With these parameters, you can pass them to
|hybridfactory.probes.custom_probe|_ like so:

.. code:: python

    # if your probe has a name
    probe = hybridfactory.probes.custom_probe(channel_map, connected, channel_positions, name)

    # alternatively, if you don't want to specify a name
    probe = hybridfactory.probes.custom_probe(channel_map, connected, channel_positions)

Be sure to ``import hybridfactory.probes`` in your ``params.py`` (see
the `example parameter file`_ to get a feel for
this).

Validation tools
----------------

For KiloSort output, we compare (shifted) templates associated with the
artificial events to templates from the sorting of the hybrid data. This
will probably be meaningless unless you use the same master file to sort
the hybrid data that you used to sort the data from which we derived our
artificial events. We
`compare`_ in one of two ways: by computing Pearson correlation coefficients of the
flattened templates (in which case, higher is better), or by computing
the Frobenius norm of the difference of the two templates (lower is
better here). When we find the best matches in a 2 ms interval around
each true firing, we can generate a `confusion matrix`_ to see how we did.

This functionality is not in a command-line tool, but should be used in a
Jupyter notebook (for now). Adding a demo notebook is a TODO.

Adding more validation tools is another TODO. Suggestions for tools
you'd want to see are `always welcome`_.

Output
------

If successful, ``hybridfactory`` will output several files in
``output_directory``:

Raw data files
~~~~~~~~~~~~~~

The filenames of your source data file will be reused, prepending ``.GT`` before
the file extension.
For example, if your source file is called ``data.bin``, the target file will be
named ``data.GT.bin`` and will live in ``output_directory``.

Dataset save files
~~~~~~~~~~~~~~~~~~

These include:

- ``metadata-[session_name].csv``: a table of filenames, start times, and sample
  rates of the files in your hybrid dataset (start times and sample rates should
  match those of your source files).
- ``annotations-[session_name].csv``: a table of (real and synthetic) cluster
  IDs, timesteps, and templates (Kilosort only) or assigned channels (JRCLUST
  only).
- ``artificial_units-[session_name].csv``: a table of new cluster IDs, true
  units, timesteps, and templates (Kilosort only) or assigned channels (JRCLUST
  only) for your artificial units.
- ``probe-[session_name].npz``: a NumPy-formatted archive of data describing
  your probe. (See `Probe configuration <#probe-configuration>`__ for a
  description of these data.)
- ``dtype-[session_name].npy``: a NumPy-formatted archive containing the sample
  rate of your dataset in the same format as your raw dataset.
- ``firings_true.npy``.
  This is a :math:`3 \times J` array of ``uint64``, where :math:`J` is the
  number of events generated.

  - Row 0 is the channel on which the event is centered, zero-based.
  - Row 1 is the timestamp of the event in sample units, zero-based.
  - Row 2 is the unit/cluster ID from the original data set for the event.

.. |hybridfactory.probes.custom_probe| replace:: ``hybridfactory.probes.custom_probe``
.. |HybridDataSet.reset| replace:: ``HybridDataSet.reset``

.. _`install Anaconda or Miniconda`: https://conda.io/docs/user-guide/install/index.html
.. _`parameter file`: #parameter-file
.. _Kilosort: https://github.com/cortex-lab/KiloSort
.. _JRCLUST: https://github.com/JaneliaSciComp/JRCLUST
.. _SpikeGLX: https://github.com/billkarsh/SpikeGLX/
.. _`validation tools`: #validation-tools
.. _`issue page`: https://gitlab.com/vidriotech/hybridfactory/issues
.. _`always welcome`: https://gitlab.com/vidriotech/hybridfactory/issues
.. _glob: https://en.wikipedia.org/wiki/Glob_%28programming%29>
.. _`NumPy data type`:  https://docs.scipy.org/doc/numpy/user/basics.types.html
.. _`natural sorting`: https://en.wikipedia.org/wiki/Natural_sort_order
.. _hybridfactory.probes.custom_probe: https://vidriotech.gitlab.io/hybridfactory/hybridfactory.probes.html#hybridfactory.probes.probe.custom_probe
.. _HybridDataSet.reset: https://vidriotech.gitlab.io/hybridfactory/hybridfactory.data.html#hybridfactory.data.dataset.HybridDataSet.reset
.. _`example parameter file`: https://vidriotech.gitlab.io/hybridfactory/#example-parameter-file
.. _`compare`: https://vidriotech.gitlab.io/hybridfactory/hybridfactory.validate.html#hybridfactory.validate.comparison.PairComparison
.. _`confusion matrix`: https://vidriotech.gitlab.io/hybridfactory/hybridfactory.validate.html#hybridfactory.validate.comparison.build_confusion_matrix
