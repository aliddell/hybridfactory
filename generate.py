#!/usr/bin/env python3

"""Copyright (C) 2018 Vidrio Technologies. All rights reserved."""

import argparse
import datetime
import importlib
import os.path as op
import shutil
import sys
import traceback

import numpy as np

import factory.io.jrc
import factory.io.raw
import factory.io.gt
import factory.generate.shift
import factory.generate.generators
from factory.io.logging import log

__author__ = "Alan Liddell <alan@vidriotech.com>"
__version__ = "0.1.0-alpha"

SPIKE_LIMIT = 25000


def _commit_hash():
    import os
    import subprocess

    old_wd = os.getcwd()
    os.chdir(op.dirname(__file__))

    try:
        ver_info = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], universal_newlines=True).strip()
    except subprocess.CalledProcessError:
        ver_info = __version__
    finally:
        os.chdir(old_wd)

    return ver_info


def _err_exit(msg, status=1):
    print(msg, file=sys.stderr)
    sys.exit(status)


def _user_dialog(msg, options=("y", "n"), default_option="n"):
    default_option = default_option.lower()
    options = [o.lower() for o in options]
    assert default_option in options

    options.insert(options.index(default_option), default_option.upper())
    options.remove(default_option)

    print(msg, end=" ")
    choice = input(f"[{'/'.join(options)}] ").strip().lower()

    iters = 0
    while choice and choice not in list(map(lambda x: x.lower(), options)) and iters < 3:
        iters += 1
        choice = input(f"[{'/'.join(options)}] ").strip().lower()

    if not choice or choice not in list(map(lambda x: x.lower(), options)):
        choice = default_option

    return choice


def _legal_params():
    required_params = {"raw_source_file": None,
                       "raw_target_file": None,
                       "data_type": [np.int16],
                       "sample_rate": None,
                       "output_type": ["kilosort", "phy"],
                       "data_directory": None,
                       "probe_type": ["npix3a", "eMouse"],
                       "ground_truth_units": None}

    optional_params = {"random_seed": None,
                       "generator_type": ["steinmetz"],
                       "num_singular_values": 6,
                       "channel_shift": None,  # depends on probe
                       "time_jitter": 50,
                       "samples_before": 40,
                       "samples_after": 40,
                       "event_threshold": -30,
                       "offset": 0,
                       "copy": False,
                       "overwrite": False}

    return required_params, optional_params


def _write_param(fh, param, param_val):
    if param == "data_type":
        if param_val == np.int16:  # no other data types supported yet
            param_val = "np.int16"
    elif isinstance(param_val, str):  # enclose string in quotes
        param_val = f'r"{param_val}"'
    elif param_val is None:
        return

    print(f"{param} = {param_val}", file=fh)


def _write_config(filename, params):
    required_params, optional_params = _legal_params()

    with open(filename, "w") as fh:
        print("import numpy as np\n", file=fh)

        print("# required parameters\n", file=fh)
        for param in required_params:
            _write_param(fh, param, params.__dict__[param])

        print("\n# optional parameters\n", file=fh)
        for param in optional_params:
            _write_param(fh, param, params.__dict__[param])

        print(f"# automatically generated on {datetime.datetime.now()}", file=fh)


def create_config(args):
    """

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments pertaining to this session.
    """

    _err_exit("This hasn't been implemented yet.", 0)


def load_params_probe(config):
    """

    Parameters
    ----------
    config : str

    Returns
    -------
    params : module
        Session parameters.
    probe : module
        Probe parameters.
    """

    params = None

    try:
        params = importlib.import_module(config)  # load parameter file as a module
    except ModuleNotFoundError:
        _err_exit(f"config file '{config}' not found")
    except SyntaxError:
        _err_exit(f"bad syntax in config file '{config}'")
    finally:
        assert params is not None

    required_params, optional_params = _legal_params()

    for param in required_params:
        if not hasattr(params, param):
            _err_exit(f"parameter '{param}' is required")

        param_val = params.__dict__[param]

        if required_params[param] is not None and param_val not in required_params[param]:
            _err_exit(f"legal values for parameter '{param}' are: {', '.join(list(map(str, required_params[param])))}")
        elif param == "raw_source_file" and not op.isfile(param_val):
            _err_exit(f"can't open source file '{param_val}'")
        elif param == "sample_rate" and (not isinstance(param_val, int) or param_val <= 0):
            _err_exit("sample_rate must be a positive integer")
        elif param == "data_directory" and not op.isdir(param_val):
            _err_exit(f"can't open data directory '{param_val}'")
        elif param == "ground_truth_units" and not hasattr(param_val, "__getitem__"):
            _err_exit("ground_truth_units must be iterable")

    probe = importlib.import_module(f"factory.probes.{params.probe_type}")  # e.g., factory.probes.npix3a

    for param in optional_params:
        if not hasattr(params, param) and param not in ("random_seed", "channel_shift"):  # set a reasonable default
            params.__dict__[param] = optional_params[param]
        elif param == "random_seed" and not hasattr(params, param):
            params.random_seed = np.random.randint(0, 2**31)
        elif param == "channel_shift" and not hasattr(params, param):
            params.channel_shift = probe.default_shift

        param_val = params.__dict__[param]

        if param not in ("generator_type", "overwrite") and not isinstance(param_val, int):
            _err_exit(f"parameter '{param}' must be an integer")
        elif param in ("overwrite", "copy") and not isinstance(param_val, bool):
            _err_exit(f"legal values for {param} are: True, False")
        elif param in ("samples_before", "samples_after", "time_jitter") and param_val <= 0:
            _err_exit(f"{param} must be a positive integer")
        elif param == "event_threshold" and param_val >= 0:
            _err_exit("event_threshold must be a negative integer")
        elif param in ("random_seed", "channel_shift", "offset") and param_val < 0:
            _err_exit(f"{param} must be a nonnegative integer")

    return params, probe


def copy_source_target(params, probe):
    """

    Parameters
    ----------
    params : module
        Session parameters.
    probe : module
        Probe parameters.

    Returns
    -------
    source : numpy.memmap
        Memory map of source data file.
    target : numpy.memmap
        Memory map of target data file.
    """

    if op.isfile(params.raw_target_file) and not params.overwrite:
        if _user_dialog(f"Target file {params.raw_target_file} exists! Overwrite?") == "y":
            params.overwrite = True
        else:
            _err_exit("aborting", 0)

    if params.copy:
        log(f"Copying {params.raw_source_file} to {params.raw_target_file}", params.verbose, in_progress=True)
        shutil.copy2(params.raw_source_file, params.raw_target_file)
        log("done", params.verbose)

    file_size_bytes = op.getsize(params.raw_source_file)
    byte_count = np.dtype(params.data_type).itemsize  # number of bytes in data type
    nrows = probe.NCHANS
    ncols = file_size_bytes // (nrows * byte_count)

    params.num_samples = ncols

    source = np.memmap(params.raw_source_file, dtype=params.data_type, offset=params.offset, mode="r",
                       shape=(nrows, ncols), order="F")
    target = np.memmap(params.raw_target_file, dtype=params.data_type, offset=params.offset, mode="r+",
                       shape=(nrows, ncols), order="F")

    return source, target


def unit_channels_union(unit_mask, params, probe):
    """

    Parameters
    ----------
    unit_mask : numpy.ndarray
        Boolean array of events to take for this unit.
    params : module
        Session parameters.
    probe : module
        Probe parameters.

    Returns
    -------
    channels : numpy.ndarray
        Channels on which unit events occur.
    """

    # select all channels on which events occur for this unit...
    event_channel_indices = factory.io.jrc.load_event_channel_indices(params.data_directory)
    channel_neighbor_indices = factory.io.jrc.load_event_channel_indices(params.data_directory)
    event_channels = probe.channel_map[probe.connected][event_channel_indices]

    # ...find neighbors for all channels...
    channel_neighbors = {}
    for channel_neighborhood in probe.channel_map[probe.connected][channel_neighbor_indices].T:
        channel_neighbors[channel_neighborhood[0]] = set(channel_neighborhood)

    # ...and isolate the channels which are neighbors of distinct centers for this unit
    unit_channel_centers = np.unique(event_channels[unit_mask])
    unit_channels = np.array(list(set.union(*[channel_neighbors[c] for c in unit_channel_centers])))

    return unit_channels


def generate_hybrid(args):
    """

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments pertaining to this session.
    """

    config_dir = op.dirname(args.config)
    config = op.basename(args.config).strip()
    if config.endswith(".py"):
        config = config[:-3]  # strip '.py' extension

    if config_dir not in sys.path:
        sys.path.insert(0, config_dir)

    params, probe = load_params_probe(config)
    params.verbose = not args.silent

    np.random.seed(params.random_seed)

    source, target = copy_source_target(params, probe)

    log("Loading event times and cluster IDs", params.verbose, in_progress=True)
    io = importlib.import_module(f"factory.io.{params.output_type}")  # e.g., factory.io.phy, factory.io.kilosort, ...
    event_times = io.load_event_times(params.data_directory)
    event_clusters = io.load_event_clusters(params.data_directory)
    log("done", params.verbose)

    gt_channels = []
    gt_times = []
    gt_labels = []

    for unit_id in params.ground_truth_units:
        unit_mask = event_clusters == unit_id
        num_events = np.where(unit_mask)[0].size
        #unit_times = event_times[unit_mask]

        if num_events > SPIKE_LIMIT:  # if more events than limit, select some to ignore
            falsify = np.random.choice(np.where(unit_mask)[0], size=num_events-SPIKE_LIMIT, replace=False)
            unit_mask[falsify] = False
            num_events = SPIKE_LIMIT
            #unit_times = np.random.choice(unit_times, size=SPIKE_LIMIT, replace=False)
        elif num_events == 0:
            log(f"No events found for unit {unit_id}", params.verbose)
            continue

        # generate artificial events for this unit
        log(f"Generating ground truth for unit {unit_id}", params.verbose, in_progress=True)

        #art_events, channel_subset = construct_artificial_events(source, event_times, unit_mask, params, probe)

        unit_times = event_times[unit_mask]
        unit_windows = factory.io.raw.unit_windows(source, unit_times, params.samples_before, params.samples_after)
        unit_windows[probe.channel_map[~probe.connected], :, :] = 0  # zero out the unconnected channels

        if params.output_type == "jrc":
            unit_channels = unit_channels_union(unit_mask, params, probe)
        else:
            unit_channels = factory.generate.generators.threshold_events(unit_windows, params.event_threshold)

        if unit_channels is None:
            log("no channels found for unit", params.verbose)
            continue

        # now create subarray for just appropriate channels
        events = unit_windows[unit_channels, :, :]  # num_channels x num_samples x num_events

        # actually generate the data
        if params.generator_type == "steinmetz":
            art_events = factory.generate.generators.steinmetz(events, params.num_singular_values)
        else:
            raise NotImplementedError(f"generator '{params.generator_type}' does not exist!")

        log("done", params.verbose)

        # shift channels
        log("Shifting channels", params.verbose, in_progress=True)
        shifted_channels = factory.generate.shift.shift_channels(unit_channels, params, probe)

        if shifted_channels is None:
            continue  # cause is logged in `shift_channels`

        log("done", params.verbose)

        # jitter events
        log("Jittering events", params.verbose, in_progress=True)
        jittered_times = factory.generate.shift.jitter_events(unit_times, params)
        log("done", params.verbose)

        if jittered_times is None:
            continue

        # write to file
        log("Writing events to file", params.verbose, in_progress=True)
        for i, jittered_center in enumerate(jittered_times):
            jittered_samples = np.arange(jittered_center - params.samples_before,
                                         jittered_center + params.samples_after + 1, dtype=jittered_center.dtype)

            shifted_window = factory.io.raw.read_roi(target, shifted_channels, jittered_samples)
            perturbed_data = shifted_window + art_events[:, :, i]

            factory.io.raw.write_roi(target, shifted_channels, jittered_samples, perturbed_data)

        log("done", params.verbose)

        cc_indices = np.abs(art_events).max(axis=1).argmax(axis=0)
        center_channels = shifted_channels[cc_indices] + 1

        gt_channels.append(center_channels)
        gt_times.append(jittered_times)
        gt_labels.append(unit_id)

    # finished writing, flush to file
    del source, target

    dirname = op.dirname(params.raw_target_file)

    # save ground-truth units for validation
    filename = factory.io.gt.save_gt_units(dirname, gt_channels, gt_times, gt_labels)
    log(f"Firing times and labels saved to {filename}.", params.verbose)

    # save parameter file for later reuse
    timestamp = int(datetime.datetime.now().timestamp())
    filename = op.join(dirname, f"params-{timestamp}.py")
    _write_config(filename, params)
    log(f"Parameter file to recreate this run saved at {filename}.", params.verbose)


def main():
    parser = argparse.ArgumentParser(description="Generate some hybrid data.")

    subparsers = parser.add_subparsers(title="optional commands")

    # auto-generate a config file
    cmd_create = subparsers.add_parser("create-config", description="create a config file")
    cmd_create.add_argument("output", nargs='?', default="params.py",
                            help="path to a config file with Python syntax (default: params.py)")
    cmd_create.set_defaults(func=create_config)

    # generate hybrid data
    cmd_generate = subparsers.add_parser("generate", description="generate some hybrid data")
    cmd_generate.add_argument("config", type=str, nargs='?', default="params.py",
                              help="path to a config file with Python syntax (default: params.py)")
    cmd_generate.add_argument("--silent", default=False, action="store_true")
    cmd_generate.set_defaults(func=generate_hybrid)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    # noinspection PyBroadException
    try:
        main()
    except Exception as e:
        err_msg = f"""A wild BUG appeared!
        
Please send the following output to {__author__}:

Version info/commit hash:
    {_commit_hash()}
Error:
    {str(e)}
Traceback:
    {traceback.format_exc()}"""
        _err_exit(err_msg)
