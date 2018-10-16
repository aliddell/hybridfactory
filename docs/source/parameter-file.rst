Parameter file
--------------

Rather than pass a bunch of flags and arguments to ``hybridfactory``, we
have collected all the parameters in a parameter file, ``params.py``.
We briefly explain each option below.
See the example, `here <#example-parameter-file>`_.

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

Example parameter file
~~~~~~~

.. literalinclude:: ../../params_example.py
   :language: python

.. |hybridfactory.probes.custom_probe| replace:: ``hybridfactory.probes.custom_probe``
.. |HybridDataSet.reset| replace:: ``HybridDataSet.reset``

.. _SpikeGLX: https://github.com/billkarsh/SpikeGLX/
.. _glob: https://en.wikipedia.org/wiki/Glob_%28programming%29>
.. _`NumPy data type`:  https://docs.scipy.org/doc/numpy/user/basics.types.html
.. _`natural sorting`: https://en.wikipedia.org/wiki/Natural_sort_order
.. _hybridfactory.probes.custom_probe: hybridfactory.probes.html#hybridfactory.probes.probe.custom_probe
.. _HybridDataSet.reset: hybridfactory.data.html#hybridfactory.data.dataset.HybridDataSet.reset
.. _`example parameter file`: #example-parameter-file
