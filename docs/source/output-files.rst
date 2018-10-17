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
