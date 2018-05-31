#!/usr/bin/env python3

# Copyright (C) 2018 Vidrio Technologies. All rights reserved.

import argparse
import datetime
import glob
import importlib
import os.path as op
import re
import shutil
import sys
import traceback

import numpy as np
import scipy.interpolate

import factory.io.gt
import factory.io.jrc
import factory.io.raw
import factory.io.spikegl
import factory.generate.shift
import factory.generate.generators

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


def _log(msg, stdout, in_progress=False):
    """

    Parameters
    ----------
    msg : str
        Message to log.
    stdout : bool
        Print to stdout if True.
    in_progress : bool, optional
        Print newline if and only if True.

    """

    end = " ... " if in_progress else "\n"

    if stdout:
        print(msg, end=end)


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
                       "from_empty": True,
                       "data_type": [np.int16],
                       "sample_rate": None,
                       "output_type": ["kilosort", "phy", "jrc"],
                       "data_directory": None,
                       "probe_type": ["npix3a", "eMouse", "hh2_arseny"],
                       "ground_truth_units": None,
                       "start_time": 0}

    optional_params = {"random_seed": None,
                       "generator_type": ["svd_generator"],
                       "num_singular_values": 6,
                       "channel_shift": None,  # depends on probe
                       "synthetic_rate": [],
                       "jitter_factor": 100,
                       "amplitude_scale_min": 0.75,
                       "amplitude_scale_max": 1.5,
                       "samples_before": 40,
                       "samples_after": 40,
                       "event_threshold": -30,
                       "offset": 0,
                       "copy": False,
                       "overwrite": False,
                       "subtract": False}

    return required_params, optional_params


def _write_param(fh, param, param_val):
    if param == "data_type":
        if param_val == np.int16:  # no other data types supported yet
            param_val = "np.int16"
    elif isinstance(param_val, str):  # enclose string in quotes
        param_val = f'r"{param_val}"'
    elif isinstance(param_val, np.ndarray):  # numpy doesn't do roundtripping
        param_val = param_val.tolist()
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

        if hasattr(required_params[param], "__getitem__") and param_val not in required_params[param]:
            _err_exit(f"legal values for parameter '{param}' are: {', '.join(list(map(str, required_params[param])))}")
        elif param == "raw_source_file":
            rsf = glob.glob(param_val)
            if not rsf:
                _err_exit(f"can't open source file '{param_val}'")
        elif param == "from_empty" and not isinstance(param_val, bool):
            _err_exit(f"parameter '{param}' must be either True or False")
        elif param == "sample_rate" and (not isinstance(param_val, int) or param_val <= 0):
            _err_exit("sample_rate must be a positive integer")
        elif param == "data_directory" and not op.isdir(param_val):
            _err_exit(f"can't open data directory '{param_val}'")
        elif param == "ground_truth_units" and not hasattr(param_val, "__len__"):
            _err_exit(f"parameter '{param}' must be iterable")
        elif param == "start_time" and param_val is not None:
            if (not isinstance(param_val, int) or param_val < 0) and not hasattr(param_val, "__len__"):
                _err_exit(f"parameter '{param}' must be a nonnegative integer, an iterable, or None")
            elif hasattr(param_val, "__len__") and any([(not isinstance(s, int) or s < 0) for s in param_val]):
                _err_exit(f"parameter '{param}', if iterable, must contain nonnegative integers")

    probe = importlib.import_module(f"factory.probes.{params.probe_type}")  # e.g., factory.probes.npix3a

    for param in optional_params:
        if not hasattr(params, param) and param not in ("random_seed", "channel_shift"):  # set a reasonable default
            params.__dict__[param] = optional_params[param]
        elif param == "random_seed" and not hasattr(params, param):
            params.random_seed = np.random.randint(0, 2 ** 31)
        elif param == "channel_shift" and not hasattr(params, param):
            params.channel_shift = None

        param_val = params.__dict__[param]

        if param not in ("generator_type", "overwrite", "amplitude_scale_min", "channel_shift",
                         "amplitude_scale_max", "synthetic_rate") and not isinstance(param_val, int):
            _err_exit(f"parameter '{param}' must be an integer")
        elif param in ("overwrite", "copy", "subtract") and not isinstance(param_val, bool):
            _err_exit(f"parameter '{param}' must be either True or False")
        elif param in ("samples_before", "samples_after", "jitter_factor") and param_val <= 0:
            _err_exit(f"parameter '{param}' must be a positive integer")
        elif param == "event_threshold" and param_val >= 0:
            _err_exit("parameter 'event_threshold' must be a negative integer")
        elif param in ("random_seed", "offset") and param_val < 0:
            _err_exit(f"parameter '{param}' must be a nonnegative integer")
        elif param.startswith("amplitude_scale") and param_val <= 0:
            _err_exit("parameter '{param}' must be a positive float")
        elif param == "firing_rate" and not hasattr(param_val, "__len__"):
            _err_exit(f"parameter '{param}' must be iterable")
        elif param == "channel_shift" and param_val is not None and param_val < 0:
            _err_exit(f"parameter '{param}' must be None or a nonnegative integer")

    if params.samples_before + params.samples_after + 1 <= np.count_nonzero(probe.connected):
        _err_exit("fewer samples than connected channels; increase samples_before, samples_after, or both")
    if params.amplitude_scale_min > params.amplitude_scale_max:
        _err_exit("amplitude_scale_min must be less than or equal to amplitude_scale_max")
    if len(params.synthetic_rate) not in (0, len(params.ground_truth_units)):
        _err_exit("synthetic rate must either be empty or specified for each ground-truth unit")
    elif any([(r <= 0 or r > params.sample_rate) for r in params.synthetic_rate]):
        _err_exit("synthetic rates must be positive integers less than the specified sample rate")

    params.me = op.abspath(config)  # save location of config file

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

    if params.raw_source_file.endswith(".bin"):
        meta_file = re.sub(r"\.bin$", ".meta", params.raw_source_file)
    else:  # .dat
        meta_file = re.sub(r"\.dat$", ".meta", params.raw_source_file)

    raw_source_files = glob.glob(params.raw_source_file)
    if params.start_time is None:
        start_times = factory.io.spikegl.get_start_times(meta_file)
    elif not hasattr(params.start_time, "__getitem__"):
        start_times = [params.start_time]
    else:
        start_times = params.start_time

    if len(raw_source_files) > 1:
        assert len(raw_source_files) == len(start_times)
        # TODO: we need a better way to rename batches of files we got from a glob
        # for now, we just add .GT to the extension and change the directory to the data_directory
        raw_target_files = raw_source_files.copy()
        for k, rtf in enumerate(raw_target_files):
            # just save hybrid data files in directory containing params file
            dirname = op.dirname(rtf)
            rtf = rtf.replace(dirname, op.dirname(params.me))

            try:
                last_dot = -(rtf[::-1].index('.') + 1)
                rtf = rtf[:last_dot] + ".GT" + rtf[last_dot:]  # add ".GT" before extension
            except ValueError:  # no '.' found in rtf
                rtf += ".GT"  # add ".GT" at the end
            finally:
                raw_target_files[k] = rtf
    else:
        raw_target_files = [params.raw_target_file]

    for k, raw_source_file in enumerate(raw_source_files):
        start_time = start_times[k]
        raw_target_file = raw_target_files[k]

        if op.isfile(raw_target_file) and not params.overwrite:
            if _user_dialog(f"Target file {raw_target_file} exists! Overwrite?") == "y":
                params.overwrite = True
            else:
                _err_exit("aborting", 0)

        if params.from_empty:
            _log(f"Laying down Gaussian noise in {raw_target_file}", params.verbose, in_progress=True)
            target = factory.io.raw.open_raw(raw_target_file, params.data_type, probe.NCHANS, mode="w+",
                                             offset=params.offset)
            factory.io.raw.lay_noise(target, 20, 65536)
            del target

            _log("done", params.verbose)
        elif params.copy:
            _log(f"Copying {raw_source_file} to {raw_target_file}", params.verbose, in_progress=True)
            shutil.copy2(raw_source_file, raw_target_file)
            _log("done", params.verbose)

        source = factory.io.raw.open_raw(raw_source_file, params.data_type, probe.NCHANS, mode="r",
                                         offset=params.offset)
        target = factory.io.raw.open_raw(raw_target_file, params.data_type, probe.NCHANS, mode="r+",
                                         offset=params.offset)

        params.num_samples = source.shape[1]

        yield source, target, start_time


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
    channel_neighbor_indices = factory.io.jrc.load_channel_neighbor_indices(params.data_directory)
    event_channels = probe.channel_map[probe.connected][event_channel_indices]

    # ...find neighbors for all channels...
    channel_neighbors = {}
    for channel_neighborhood in probe.channel_map[probe.connected][channel_neighbor_indices].T:
        channel_neighbors[channel_neighborhood[0]] = set(channel_neighborhood)

    # ...and isolate the channels which are neighbors of distinct centers for this unit
    unit_channel_centers = np.unique(event_channels[unit_mask])
    unit_channels = np.array(list(set.union(*[channel_neighbors[c] for c in unit_channel_centers])))

    return unit_channels


def scale_events(events, params):
    """

    Parameters
    ----------
    events : numpy.ndarray
        Tensor, num_channels x num_samples x num_events.
    params : module
        Session parameters.

    Returns
    -------
    scaled_events : numpy.ndarray
        Tensor, num_channels x num_samples x num_events, scaled.
    """

    abs_events = np.abs(events)

    centers = abs_events.max(axis=0).argmax(axis=0)

    scale_factors = np.random.uniform(params.amplitude_scale_min, params.amplitude_scale_max, size=abs_events.shape[2])
    scale_rows = [np.hstack((np.linspace(0, scale_factors[i], centers[i]),
                             np.linspace(scale_factors[i], 0, events.shape[1] - centers[i] + 1)[1:]))[np.newaxis, :] for
                  i in range(events.shape[2])]

    return np.stack(scale_rows, axis=2) * events


def remove_spike(event_window, params):
    """Remove spike from window via spline interpolation.

    Parameters
    ----------
    event_window : numpy.ndarray
        Region in raw data centered around a spiking event.
    params : module
        Session parameters.
    Returns
    -------
    new_window : numpy.ndarray
        Copy of region with spike removed.
    """

    def _find_left_right(win, center=None):
        assert win.ndim == 1

        m = win.size // 2
        rad = win.size // 8

        if center is None:  # assume window is "centered" at event
            nbd = win[m - rad:m + rad]
            center = np.abs(nbd).argmax() + m - rad
            is_local_min = nbd[rad] == nbd.min()
        else:
            nbd = win[center - rad:center + rad]
            is_local_min = nbd[rad] == nbd.min()

        # find points where first and second derivative change signs
        wdiff = np.diff(win)  # first derivative
        scale_factor = np.abs(wdiff)
        scale_factor[scale_factor == 0] = 1
        abswdiff = wdiff / scale_factor

        wdiff2 = np.diff(wdiff)  # second derivative
        scale_factor2 = np.abs(wdiff2)
        scale_factor2[scale_factor2 == 0] = 1
        abswdiff2 = wdiff2 / scale_factor2

        turning_points = np.union1d(
                np.where(np.array([abswdiff[i] != abswdiff[i + 1] for i in range(abswdiff.size - 1)]))[0] + 1,
                np.where(np.array([abswdiff2[i] != abswdiff2[i + 1] for i in range(abswdiff2.size - 1)]))[0] + 2)
        tp_center = np.abs(center - turning_points).argmin()

        if tp_center < 2 or tp_center > turning_points.size - 2:
            if is_local_min:  # all differences are negative until center, then positive
                # find the last difference before the center that is positive
                wleft = center - np.where(wdiff[:center - 1][::-1] > 0)[0][0]
                # find the first difference after the center that is negative
                wright = center + np.where(wdiff[center + 1:] < 0)[0][0]
            else:  # all differences are positive until center, then negative
                # find the last difference before the center that is negative
                wleft = center - np.where(wdiff[:center - 1][::-1] < 0)[0][0]
                # find the first difference after the center that is positive
                wright = center + np.where(wdiff[center + 1:] > 0)[0][0]
        else:
            wleft = turning_points[tp_center - 2] + (center - turning_points[tp_center - 2]) // 2
            wright = turning_points[tp_center + 2] + (turning_points[tp_center + 2] - center) // 2

        return wleft, wright

    event_center = params.samples_before

    new_window = event_window.copy().astype(np.float64)
    for k, window in enumerate(event_window):
        wl, wr = _find_left_right(window, center=event_center)

        exes = np.hstack((np.arange(wl), np.arange(wr, window.size)))
        whys = window[exes]
        g = scipy.interpolate.interp1d(exes, whys, "cubic")

        new_window[k, :] = g(np.arange(window.size))
        new_window[k, wl:wr] += 3 * np.random.randn(wr - wl)

    return new_window


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

    _log("Loading event times and cluster IDs", params.verbose, in_progress=True)
    io = importlib.import_module(f"factory.io.{params.output_type}")  # e.g., factory.io.phy, factory.io.kilosort, ...
    event_times = io.load_event_times(params.data_directory)
    event_clusters = io.load_event_clusters(params.data_directory)
    _log("done", params.verbose)

    # data for firings_true.npy
    gt_channels = []
    gt_times = []
    gt_labels = []

    for source, target, start_time in copy_source_target(params, probe):
        time_mask = (event_times - params.samples_before >= start_time) & (event_times - start_time +
                                                                           params.samples_after < target.shape[1])

        for k, unit_id in enumerate(params.ground_truth_units):
            unit_mask = (event_clusters == unit_id) & time_mask
            num_events = np.where(unit_mask)[0].size

            if num_events > SPIKE_LIMIT:  # if more events than limit, select some to ignore
                falsify = np.random.choice(np.where(unit_mask)[0], size=num_events - SPIKE_LIMIT, replace=False)
                unit_mask[falsify] = False
            elif num_events == 0:
                _log(f"No events found for unit {unit_id}", params.verbose)
                continue

            # generate artificial events for this unit
            _log(f"Generating ground truth for unit {unit_id}", params.verbose, in_progress=True)

            unit_times = event_times[unit_mask] - start_time
            unit_windows = factory.io.raw.unit_windows(source, unit_times, params.samples_before,
                                                       params.samples_after,
                                                       car_channels=np.where(probe.connected)[0])
            unit_windows[probe.channel_map[~probe.connected], :, :] = 0  # zero out the unconnected channels

            if params.output_type == "jrc":
                unit_channels = unit_channels_union(unit_mask, params, probe)
            else:
                unit_channels = factory.generate.generators.threshold_events(unit_windows, params.event_threshold)

            if unit_channels is None:
                _log("no channels found for unit", params.verbose)
                continue

            # now create subarray for just appropriate channels
            events = unit_windows[unit_channels, :, :]  # num_channels x num_samples x num_events

            # actually generate the data
            if params.generator_type == "svd_generator":
                if num_events < params.num_singular_values:
                    _log("not enough events to generate!", params.verbose)
                    continue
                art_events = factory.generate.generators.svd_generator(events, params.num_singular_values)
            else:
                raise NotImplementedError(f"generator '{params.generator_type}' does not exist!")

            art_events = scale_events(art_events, params)
            _log("done", params.verbose)

            # shift channels
            _log("Shifting channels", params.verbose, in_progress=True)
            shifted_channels = factory.generate.shift.shift_channels(unit_channels, params, probe)

            if shifted_channels is None:
                _log("failed!", params.verbose)
                continue

            _log("done", params.verbose)

            # jitter events
            _log("Jittering events", params.verbose, in_progress=True)
            if params.synthetic_rate:
                synthetic_rate = params.synthetic_rate[k]
                stop = target.shape[1] - params.samples_before - params.samples_after - 1
                step = params.sample_rate//synthetic_rate
                synthetic_times = params.samples_before + 1 + np.arange(start=0, stop=stop, step=step)
                jittered_times = factory.generate.shift.jitter_events(synthetic_times, params.sample_rate,
                                                                      params.jitter_factor, params.samples_before,
                                                                      params.samples_after, params.num_samples)
            else:
                jittered_times = factory.generate.shift.jitter_events(unit_times, params.sample_rate,
                                                                      params.jitter_factor, params.samples_before,
                                                                      params.samples_after, params.num_samples)
            _log("done", params.verbose)

            # sample artificial events with replacement
            art_events = art_events[:, :, np.random.choice(art_events.shape[2], size=jittered_times.size)]

            # write to file
            _log("Writing events to file", params.verbose, in_progress=True)
            for i, jittered_center in enumerate(jittered_times):
                # first subtract the true event
                if params.subtract:
                    event_center = event_times[i]
                    event_samples = np.arange(event_center - params.samples_before,
                                              event_center + params.samples_after + 1, dtype=event_center.dtype)
                    event_window = factory.io.raw.read_roi(target, unit_channels, event_samples)

                    subtracted_window = remove_spike(event_window, params)
                    factory.io.raw.write_roi(target, unit_channels, event_samples, subtracted_window)

                # now add the shifted event
                jittered_samples = np.arange(jittered_center - params.samples_before,
                                             jittered_center + params.samples_after + 1, dtype=jittered_center.dtype)

                shifted_window = factory.io.raw.read_roi(target, shifted_channels, jittered_samples)
                perturbed_data = shifted_window + art_events[:, :, i]

                factory.io.raw.write_roi(target, shifted_channels, jittered_samples, perturbed_data)

            _log("done", params.verbose)

            cc_indices = np.abs(art_events).max(axis=1).argmax(axis=0)
            center_channels = shifted_channels[cc_indices]

            gt_channels.append(center_channels)
            gt_times.append(jittered_times + start_time)
            gt_labels.append(unit_id)

        # finished writing, flush to file
        del source, target

    # save everything for later
    dirname = op.dirname(params.me)

    # save ground-truth units for validation
    filename = factory.io.gt.save_gt_units(dirname, gt_channels, gt_times, gt_labels)
    _log(f"Firing times and labels saved to {filename}.", params.verbose)

    # save parameter file for later reuse
    timestamp = int(datetime.datetime.now().timestamp())
    filename = op.join(dirname, f"params-{timestamp}.py")
    _write_config(filename, params)
    _log(f"Parameter file to recreate this run saved at {filename}.", params.verbose)


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
