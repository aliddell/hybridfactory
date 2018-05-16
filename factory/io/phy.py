"""Copyright (C) 2018 Vidrio Technologies. All rights reserved."""

import os.path as op

import numpy as np


def _read_npy(filename, flatten=True):
    """

    Parameters
    ----------
    filename : str
        Path to file to load.
    flatten : bool, optional
        Convert to 1d array.

    Returns
    -------
    result : numpy.ndarray
    """

    assert op.isfile(filename)
    result = np.load(filename)

    if flatten:
        result = result.ravel()

    return result


def load_event_templates(dirname):
    """

    Parameters
    ----------
    dirname : str
        Path to directory containing file to load.

    Returns
    -------
    result : numpy.ndarray
    """

    filename = op.join(dirname, "spike_templates.npy")
    return _read_npy(filename)


def load_templates(dirname):
    """

    Parameters
    ----------
    dirname : str
        Path to directory containing file to load.

    Returns
    -------
    result : numpy.ndarray
    """

    filename = op.join(dirname, "templates.npy")
    return _read_npy(filename, flatten=False)


def load_event_times(dirname):
    """

    Parameters
    ----------
    dirname : str
        Path to directory containing file to load.

    Returns
    -------
    result : numpy.ndarray
    """

    filename = op.join(dirname, "spike_times.npy")
    result = _read_npy(filename)
    assert (result == np.sort(result)).all()

    return result


def load_event_clusters(dirname):
    """

    Parameters
    ----------
    dirname : str
        Path to directory containing file to load.

    Returns
    -------
    result : numpy.ndarray
    """

    filename = op.join(dirname, "spike_clusters.npy")
    return _read_npy(filename).astype(np.int64)

