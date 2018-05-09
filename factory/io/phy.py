import os.path as op

import numpy as np


def _read_npy(filename, shift=False, flatten=True):
    """

    Parameters
    ----------
    filename : str
        Path to file to load.
    shift : bool, optional
        Subtract 1 from values to account for 1-based indexing by Matlab.
    flatten : bool, optional

    Returns
    -------
    result : numpy.ndarray
    """

    assert op.isfile(filename)
    result = np.load(filename)

    if shift:
        result = result - 1
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
    result = _read_npy(filename, shift=True).astype(np.int64)
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
    return _read_npy(filename)

