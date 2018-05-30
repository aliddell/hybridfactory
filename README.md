# Hybrid ground-truth generation for spike-sorting

## Installation

The best way to get started is to [install Anaconda or Miniconda](https://conda.io/docs/user-guide/install/index.html).
Once you've done that, fire up your favorite terminal emulator (PowerShell or CMD on Windows; iTerm2 or Terminal on Mac;
lots of choices if you're on Linux, but you knew that) and navigate to the directory containing this README file (it
also contains `environment.yml`).

On UNIX variants, type:

```bash
$ conda env create -n hybridfactory
Solving environment: done
Downloading and Extracting Packages
...
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use:
# > source activate hybridfactory
#
# To deactivate an active environment, use:
# > source deactivate
#

$ source activate hybridfactory
```

On Windows:

```powershell
$ conda env create -n hybridfactory
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
```

and you should be good to go. Remember that `[source] activate hybridfactory` every time you open up a new shell.

## Usage

This tool is primarily a command-line utility.
Provided you have a [parameter file](#parameter-file), you can invoke it like so:

```bash
$ /path/to/python3 generate.py generate /path/to/params.py
```

Right now, `generate` is the only command available, allowing you to generate hybrid data from a pre-existing raw
data set and output from a spike-sorting tool, e.g., [KiloSort](https://github.com/cortex-lab/KiloSort) or
[JRCLUST](https://github.com/JaneliaSciComp/JRCLUST).
This is probably what you want to do.

After your hybrid data has been generated, we have some [validation tools](#validation-tools) you can use to look at
your hybrid output, but this is not as convenient as a command-line tool (yet).

### A word about bugs

This software is under active development.
Although we strive for accuracy and consistency, there's a good chance you'll run into some bugs.
If you run into an exception which crashes the program, you should see a helpful message with my email address and a
traceback.
If you find something a little more subtle, please post an issue on the
[issue page](https://gitlab.com/vidriotech/spiegel/hybridfactory/issues).

## Parameter file

Rather than pass a bunch of flags and arguments to `generate.py`, we have collected all the parameters in a
parameter file, `params.py`.
We briefly explain each option below.
See [sample_params.py](https://gitlab.com/vidriotech/spiegel/hybridfactory/blob/master/sample_params.py) for an example.

### Required parameters

- `raw_source_file`: Path to file containing raw source data (currently only SpikeGL-formatted data is supported).
  This can also be a [glob](https://en.wikipedia.org/wiki/Glob_%28programming%29) if you have multiple data files.
- `raw_target_file`: Path to file which will contain hybrid data.
  If you use a glob in `raw_source_file`, this will be ignored and hybrid data will be created in the folder containing
   `params.py` and with a similar name scheme as the source data.
   The only difference is a `.GT` just before the file extension.
- `from_empty`: Whether or not to start from an empty file (with some Gaussian background noise).
  If you just want your ground-truth units without any other events in the background, set this to True. 
- `data_type`: Type of raw data, as a [NumPy data type](https://docs.scipy.org/doc/numpy/user/basics.types.html).
  As of this writing, I have only seen `int16`.
- `sample_rate`: Sample rate of the source data, in Hz.
- `data_directory`: Directory containing output from your spike sorter, e.g., `rez.mat` or `*.npy` for KiloSort;
  or `*_jrc.mat` and `*_spk(raw|wav|fet).jrc` for JRCLUST.
- `output_type`: Type of output from your spike sorter.
  One of "phy" (for `*.npy`), "kilosort" (for `rez.mat`), or "jrc" (for `*_jrc.mat` and `*_spk(raw|wav|fet).jrc`).
- `probe_type`: Probe layout.
  One of "npix3a" (for [Neuropixels Phase 3A](https://github.com/cortex-lab/neuropixels/wiki/About_Neuropixels)),
  "hh2_arseny" (for
  [this JRCLUST probe layout](https://github.com/JaneliaSciComp/JRCLUST/blob/master/prb/hh2_arseny.prb)), or "eMouse"
  (for the synthetic [eMouse](https://github.com/cortex-lab/KiloSort/tree/master/eMouse) example data from KiloSort).
- `ground_truth_units`: Indices (i.e., cluster labels) of ground-truth units from your spike sorter's output.
  Zero-indexed.
- `start_time`: Start time of data file in sample units.
  Nonnegative integer if `raw_source_file` is a single file, iterable of nonnegative integers if you have a globbed
  `raw_source_file`.
  Can also be `None` if you have meta files named similarly to your data files.

### Optional parameters

- `random_seed`: Nonnegative integer in the range $`[0, 2^{31})`$.
  Because this algorithm is randomized, setting a random seed allows for reproducible output.
  The default is itself randomly generated, but will be output in a `params-[TIMESTAMP].py` on successful completion.
- `generator_type`: Algorithm used to construct the artificial events.
  By default, we use a variant of [this approach](https://github.com/cortex-lab/groundTruth).
- `num_singular_values`: Number of singular values to use in the construction of artificial events.
  Default is 6.
- `channel_shift`: Number of channels to shift artificial events up or down from their source.
  Default depends on the probe used.
- `synthetic_rate`: Firing rate, in Hz, for hybrid units.
  This should be either an empty list (if you want to use the implicit firing rate of your ground-truth units) or an
  iterable of artificial rates.
  In the latter case, you must specify a firing rate for each ground-truth unit (this is the default behavior).
- `time_jitter`: Scale factor for (normally-distributed) random time shift, in sample units.
  Default is 100.
- `amplitude_scale_min`: Minimum factor for (uniformly-distributed) random amplitude scaling, in percentage units.
  Default is 0.75.
- `amplitude_scale_min`: Maximum factor for (uniformly-distributed) random amplitude scaling, in percentage units.
  Default is 1.5.
- `samples_before`: Number of samples to take before an event timestep for artificial event construction.
  Default is 40.
- `samples_after`: Number of samples to take after an event timestep for artificial event construction.
  Default is 40.
- `event_threshold`: Negative threshold a channel must exceed to be considered part of an event.
  Default is -30.
- `offset`: Point in the raw file at which the data starts.
  Default is 0.
- `copy`: Whether or not to copy the source file to the target.
  You usually want to do this, but if the file is large and you know where your data has been perturbed, you could use
  [`reset_target`](https://gitlab.com/vidriotech/spiegel/hybridfactory/blob/master/factory/io/raw.py#L102) instead.
  Default is False.
- `overwrite`: Whether or not to overwrite a target file if it already exists.
  If this is False, you will be prompted.
  Default is False.
- `subtract`: Whether or not to try to remove your ground-truth units from their original locations before you shift
  them.
  This is an experimental feature, and may leave artifacts.
  Default is False. 

## Validation tools

For KiloSort output, we compare (shifted) templates associated with the artificial events to templates from the sorting
of the hybrid data.
This will probably be meaningless unless you use the same master file to sort the hybrid data that you used to sort the
data from which we derived our artificial events.
We [compare](https://gitlab.com/vidriotech/spiegel/hybridfactory/blob/master/factory/validate/template.py#L52) in one of
two ways: by computing correlation scores of the templates (in which case, higher is better), or by computing the
(Frobenius) norm of the difference of the two templates (lower is better here).
When we find the best matches in a 2 ms interval around each true firing, we can generate a
[confusion matrix](https://gitlab.com/vidriotech/spiegel/hybridfactory/blob/master/factory/validate/comparison.py#L5)
to see how we did.

This functionality is not in `generate.py`, but should be used in a Jupyter notebook (for now).

Adding more validation tools is a TODO.
Suggestions for tools you'd want to see are
[always welcome](https://gitlab.com/vidriotech/spiegel/hybridfactory/issues).
