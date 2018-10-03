# Copyright (C) 2018 Vidrio Technologies. All rights reserved.

import glob
from os import path as op
import re

import numpy as np

from hybridfactory.utils import natsort


def get_start_times(filename):
    """Load start times (in units of samples) from a SpikeGL meta file (or glob of same).

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
            line = fh.readline().strip()
            while line and not line.lower().startswith("firstsample"):
                line = fh.readline().strip()

            if not line:
                raise IOError(f"file {fn} does not have a firstSample field")

        return int(re.split(r"\s*=\s*", line)[1])

    if isinstance(filename, str):
        filenames = glob.glob(filename)
        if not filenames:
            raise IOError(f"no such file or bad glob: {filename}")
    elif hasattr(filename, "__iter__"):
        filenames = list(filename)
        if not all([op.isfile(fn) for fn in filenames]):
            raise IOError("missing files")

    sorted_files = natsort(filenames)

    # get absolute start time
    t0 = _find_start(sorted_files[0])

    return np.array([_find_start(f) - t0 for f in sorted_files])
