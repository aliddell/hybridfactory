# Copyright (C) 2018 Vidrio Technologies. All rights reserved.

import glob
from os import path as op
import re

import numpy as np


def _natsort(strings):
    ttime = re.compile(r"_t(\d+)\.")

    def _natkey(term):
        res = ttime.search(term)
        return int(res.group(1)) if res is not None else term

    return sorted(strings, key=_natkey)


def get_start_times(filename):
    """Load start times from a SpikeGL meta file (or glob of same).

    Parameters
    ----------
    filename : str
        Path or glob to .meta files.

    Returns
    -------
    start_times : numpy.ndarray
        Start times in order of glob expansion, zero-indexed.

    """
    def _find_start(fn):
        with open(fn, "r") as fh:
            lines = [l.strip() for l in fh.readlines()]
            start_time = [int(re.split(r"\s*=\s*", l)[1]) for l in lines if l.lower().startswith("firstsample")]
            if len(start_time) < 1:
                raise IOError(f"file {fn} does not have a firstSample field")

        return start_time[0]

    if isinstance(filename, str):
        filenames = glob.glob(filename)
        if not filenames:
            raise IOError(f"no such file or bad glob: {filename}")
    elif hasattr(filename, "__iter__"):
        filenames = list(filename)
        if not all([op.isfile(fn) for fn in filenames]):
            raise IOError("missing files")

    sorted_files = _natsort(filenames)

    # get absolute start time
    t0 = _find_start(sorted_files[0])

    return np.array([_find_start(f) - t0 for f in filenames])
