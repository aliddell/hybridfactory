#!/usr/bin/env python3

# Copyright (C) 2018 Vidrio Technologies. All rights reserved.

import argparse
import datetime
import glob
import hashlib
import importlib
import numbers
import os.path as op
import sys
import traceback

import numpy as np

import hybridfactory.data.annotation
import hybridfactory.data.dataset
import hybridfactory.io.spikegl
import hybridfactory.generate.generator
from hybridfactory.generate.generator import NoEventException, RankDeficientException
from hybridfactory.probes.probe import Probe

__author__ = "Alan Liddell <alan@vidriotech.com>"
__version__ = "0.1.0-beta"

SPIKE_LIMIT = 25000


def _commit_hash():
    import os
    import subprocess

    old_wd = os.getcwd()
    os.chdir(op.dirname(op.dirname(__file__)))

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
    required_params = {"data_directory": None,                           # path to a directory
                       "raw_source_file": None,                          # path to a file, or a glob
                       "data_type": tuple(set(np.sctypeDict.values())),  # NumPy dtype
                       "sample_rate": None,                              # positive int
                       "ground_truth_units": None,                       # iterable of int
                       "start_time": 0,                                  # nonnegative int
                       "probe": None}                                    # Probe

    optional_params = {"session_name": None,                             # str
                       "random_seed": np.random.randint(0, 2**31),       # int in [0, 2^31)
                       "output_directory": None,                         # path to a file
                       "output_type": None,                              # (phy, kilosort, jrc)
                       "num_singular_values": 6,                         # positive int
                       "channel_shift": None,                            # int
                       "synthetic_rate": [],                             # iterable of positive int
                       "jitter_factor": 100,                             # nonnegative int
                       "amplitude_scale_min": 1.,                        # positive float
                       "amplitude_scale_max": 1.,                        # positive float
                       "samples_before": 40,                             # positive int
                       "samples_after": 40,                              # positive int
                       "event_threshold": -30,                           # negative int
                       "copy": False,                                    # bool
                       "erase": False}                                   # bool

    return required_params, optional_params


def _write_probe(fh, probe):
    print(f"channel_map = np.array({probe.channel_map.tolist()})", file=fh)
    print(f"connected = np.array({probe.channel_map.tolist()})", file=fh)
    print(f"xcoords = np.array({probe.channel_positions[:, 0].tolist()})".replace("nan", "np.nan"), file=fh)
    print(f"ycoords = np.array({probe.channel_positions[:, 1].tolist()})".replace("nan", "np.nan"), file=fh)
    print(f"channel_positions = np.hstack((xcoords[:, np.newaxis], ycoords[:, np.newaxis]))", file=fh)
    print(f'probe_name = r"{probe.name}"', file=fh)
    print(f"probe = custom_probe(channel_map, connected, channel_positions, probe_name)", file=fh)


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
        print("import numpy as np", file=fh)
        print("from hybridfactory.probes import custom_probe\n", file=fh)

        print("# REQUIRED PARAMETERS\n", file=fh)
        for param in required_params:
            if param != "probe":
                _write_param(fh, param, params.__dict__[param])

        print("\n# PROBE CONFIGURATION\n", file=fh)
        _write_probe(fh, params.probe)

        print("\n# OPTIONAL PARAMETERS\n", file=fh)
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


def load_params(config):
    """

    Parameters
    ----------
    config : str

    Returns
    -------
    params : module
        Session parameters.
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

        if isinstance(required_params[param], tuple) and param_val not in required_params[param]:
            _err_exit(f"legal values for parameter '{param}' are: {', '.join(list(map(str, required_params[param])))}")
        elif param == "data_directory" and not op.isdir(param_val):
            _err_exit(f"can't open data directory '{param_val}'")
        elif param == "raw_source_file":
            rsf = glob.glob(param_val)
            if not rsf:
                _err_exit(f"can't open source file '{param_val}'")
        elif param == "sample_rate" and (not isinstance(param_val, numbers.Integral) or param_val <= 0):
            _err_exit("sample_rate must be a positive integer")
        elif param == "ground_truth_units" and not hasattr(param_val, "__iter__"):
            _err_exit(f"parameter '{param}' must be iterable")
        elif param == "start_time":
            if not hasattr(param_val, "__iter__"):
                param_val = np.array([param_val])
            else:
                param_val = np.array(param_val)

            if not np.issubdtype(param_val.dtype, np.integer) or (param_val < 0).any():
                _err_exit(f"parameter '{param}', if iterable, must contain nonnegative integers")
        elif param == "probe" and not isinstance(param_val, Probe):
            _err_exit(f"parameter '{param}' must be an instance of Probe")

    for param in optional_params:
        if not hasattr(params, param):  # set a reasonable default
            params.__dict__[param] = optional_params[param]

        param_val = params.__dict__[param]

        # check types
        if param in ("random_seed", "num_singular_values", "jitter_factor", "samples_before", "samples_after",
                     "event_threshold") and not isinstance(param_val, numbers.Integral):
            _err_exit(f"parameter '{param}' must be an integer")
        elif param == "channel_shift" and param_val is not None and not isinstance(param_val, numbers.Integral):
            _err_exit(f"parameter '{param}' must be None or a nonnegative integer")
        elif param.startswith("amplitude_scale") and not isinstance(param_val, numbers.Real):
            _err_exit(f"parameter '{param}' must be a float")
        elif param in ("copy", "erase") and not isinstance(param_val, bool):
            _err_exit(f"parameter '{param}' must be a bool")
        elif param in ("output_type", "output_directory",
                       "session_name") and param_val is not None and not isinstance(param_val, str):
            _err_exit(f"parameter '{param}' must be a string")
        elif param == "synthetic_rate" and not hasattr(param_val, "__iter__"):
            _err_exit(f"parameter '{param}' must be iterable")

        # check values
        if param == "random_seed" and (param_val < 0 or param_val >= 2**31):
            _err_exit(f"parameter '{param}' must be an integer between 0 and {2**31}")
        elif param in ("num_singular_values", "samples_before", "samples_after") and param_val <= 0:
            _err_exit(f"parameter '{param}' must be a positive integer")
        elif param == "jitter_factor" and param_val < 0:
            _err_exit(f"parameter '{param}' must be a nonnegative integer")
        elif param == "event_threshold" and param_val >= 0:
            _err_exit(f"parameter '{param}' must be a negative integer")
        elif param.startswith("amplitude_scale") and param_val <= 0:
            _err_exit("parameter '{param}' must be a positive float")
        elif param == "channel_shift" and param_val is not None and param_val < 0:
            _err_exit(f"parameter '{param}' must be None or a nonnegative integer")
        elif param == "session_name" and param_val == "":
            _err_exit(f"parameter '{param}' must be a nonempty string")
        elif param == "output_type" and param_val not in ("phy", "kilosort", "jrc"):
            _err_exit(f"parameter '{param}' must be one of 'phy', 'kilosort', or 'jrc'")

    params.ground_truth_units = np.array(params.ground_truth_units)
    params.synthetic_rate = np.array(params.synthetic_rate)
    if params.output_directory is None:
        params.output_directory = op.join(params.data_directory, "hybrid_output")
    if params.session_name is None:
        params.session_name = hashlib.md5(str(datetime.datetime.now().timestamp()).encode()).hexdigest()[-8:]

    if params.amplitude_scale_min > params.amplitude_scale_max:
        _err_exit("amplitude_scale_min must be less than or equal to amplitude_scale_max")
    if params.synthetic_rate.size not in (0, params.ground_truth_units.size):
        _err_exit("synthetic rate must either be empty or specified for each ground-truth unit")
    if (params.synthetic_rate <= 0).any():
        _err_exit("synthetic rates must be positive integers less than the specified sample rate")

    params.me = op.abspath(config)  # save location of config file

    return params


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

    params = load_params(config)
    params.verbose = not args.silent

    probe = params.probe
    np.random.seed(params.random_seed)

    # load up the source data set
    _log("Loading source dataset", params.verbose, in_progress=True)
    try:
        source = hybridfactory.data.dataset.new_annotated_dataset(filename=params.raw_source_file,
                                                            dtype=params.data_type,
                                                            sample_rate=params.sample_rate,
                                                            probe=probe,
                                                            ann_location=params.data_directory,
                                                            ann_format=params.output_type)
    except (TypeError, ValueError) as err:
        _err_exit(f"Error: {str(err)}")
    _log("done", params.verbose)

    # open read-only
    source.open_raw("r")

    # create the hybrid data set
    _log("Creating target dataset", params.verbose, in_progress=True)
    try:
        target = hybridfactory.data.dataset.new_hybrid_dataset(source, output_directory=params.output_directory,
                                                         copy=params.copy, create=True, transform=None)
    except (TypeError, ValueError) as err:
        _err_exit(f"Error: {str(err)}")
    _log("done", params.verbose)

    # open for read-write
    target.open_raw("r+")

    # create the hybrid data generator
    generator = hybridfactory.generate.generator.SVDGenerator(dataset=target,
                                                        event_threshold=params.event_threshold,
                                                        samples_before=params.samples_before,
                                                        samples_after=params.samples_after)

    for k, unit_id in enumerate(params.ground_truth_units):
        # generate artificial events for this unit
        _log(f"Generating ground truth for unit {unit_id}", params.verbose)

        # find the channels for this unit according to our threshold
        _log(f"\tFinding channels to shift", params.verbose, in_progress=True)
        unit_channels = target.unit_channels(unit=unit_id,
                                             threshold=params.event_threshold,
                                             samples_before=params.samples_before,
                                             samples_after=params.samples_after)
        if unit_channels.size == 0:
            _log(f"no channels found for this unit!", params.verbose)
            continue
        else:
            _log("done", params.verbose)

        # find channel shift
        _log(f"\tFinding new channels", params.verbose, in_progress=True)
        shifted_channels = generator.shift_channels(channels=unit_channels,
                                                    shift_factor=params.channel_shift)
        if np.isnan(shifted_channels).any():
            _log(f"no channel shift found for this unit!", params.verbose)
            continue
        else:
            _log("done", params.verbose)

        _log("\tFinding new times", params.verbose, in_progress=True)
        if params.synthetic_rate.size > 0:  # generate synthetic firing times
            rate = params.synthetic_rate[k]

            firing_times = generator.synthetic_firing_times(firing_rate=rate,
                                                            jitter_factor=params.jitter_factor)
        else:  # use initial event firing times and jitter them
            firing_times = generator.jitter_events(event_times=source.unit_firing_times(unit_id),
                                                   jitter_factor=params.jitter_factor)

        if firing_times.size == 0:
            _log(f"no times found for this unit!", params.verbose)
            continue
        else:
            firing_times = np.random.choice(firing_times, size=min(firing_times.size, SPIKE_LIMIT),
                                            replace=False)
            _log("done", params.verbose)

        # actually construct the events
        _log(f"\tConstructing artificial events", params.verbose, in_progress=True)
        try:
            artificial_events = generator.construct_events(unit=unit_id,
                                                           num_singular_values=params.num_singular_values,
                                                           num_events=min(firing_times.size, SPIKE_LIMIT))
        except (NoEventException, RankDeficientException):  # not enough events to generate
            _log("not enough events to work with!", params.verbose)
            continue

        _log("done", params.verbose)

        # scale them
        _log(f"\tScaling artificial events", params.verbose, in_progress=True)
        artificial_events = generator.scale_events(events=artificial_events,
                                                   scale_min=params.amplitude_scale_min,
                                                   scale_max=params.amplitude_scale_max)

        _log("done", params.verbose)

        if params.erase:  # erase unit
            _log("\tErasing original unit", params.verbose, in_progress=True)
            generator.erase_unit(unit=unit_id,
                                 kind="cubic")
            _log("done", params.verbose)

        _log("\tInserting artificial unit", params.verbose, in_progress=True)
        generator.insert_unit(events=artificial_events,
                              event_times=firing_times,
                              channels=shifted_channels,
                              true_unit=unit_id)
        _log("done", params.verbose)

    source.close_raw()
    target.close_raw()

    # save everything for later
    dirname = params.output_directory
    session = params.session_name

    hybridfactory.data.dataset.save_dataset(target, dirname, session)

    # also save firings matrix
    target.export_ground_truth_matrix(op.join(dirname, f"firings_true.npy"))

    _log(f"Session information saved in {dirname}.", params.verbose)

    # save parameter file for later reuse
    filename = op.join(dirname, f"hfparams_{session}.py")
    _write_config(filename, params)
    _log(f"Parameter file to recreate this run saved at {filename}.", params.verbose)


def _main():
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


def main():
    if __name__ == "__main__":
        # noinspection PyBroadException
        try:
            _main()
        except Exception as e:
            err_msg = f"""This looks like a bug!

    Please send the following output to {__author__}:

    Version info/commit hash:
        {_commit_hash()}
    Error:
        {str(e)}
    Traceback:
        {traceback.format_exc()}"""
            _err_exit(err_msg)
