"""Copyright (C) 2018 Vidrio Technologies. All rights reserved."""

import os
import os.path as op

import h5py
import numpy as np
import scipy.io


def _read_matlab(dirname, field, flatten=False):
    """

    Parameters
        ----------
    dirname : str
        Path to directory containing *_jrc.mat.
    field : str
        Field to read from MAT file.

    Returns
    -------
    result : numpy.ndarray
    """

    assert op.isdir(dirname)
    filename = op.join(dirname, [f for f in os.listdir(dirname) if f.endswith("_jrc.mat")][0])

    try:
        matfile = scipy.io.loadmat(filename)
    except NotImplementedError:
        matfile = h5py.File(filename, 'r')

    if isinstance(matfile, dict):  # old-style MAT file
        result = matfile[field]
    else:
        result = matfile.get(field).value
        matfile.close()

    if flatten:
        result = result.ravel()
    else:
        result = result.T

    return result


def _read_jrc(dirname, dtype):
    """

    Parameters
    ----------
    dirname : str
        Path to directory containing file to load.
    dtype : {"raw", "filtered", "features"}

    Returns
    -------
    result : numpy.ndarray
    """

    assert dtype in ("raw", "filtered", "features")

    if dtype == "raw":
        dims = tuple(_read_matlab(dirname, "dimm_raw", flatten=True).astype(np.int32))
        dtype = np.int16
        suffix = "_spkraw.jrc"
    elif dtype == "filtered":
        dims = tuple(_read_matlab(dirname, "dimm_spk", flatten=True).astype(np.int32))
        dtype = np.int16
        suffix = "_spkwav.jrc"
    else:  # dtype == "features"
        dims = tuple(_read_matlab(dirname, "dimm_fet", flatten=True).astype(np.int32))
        dtype = np.float32
        suffix = "_spkfet.jrc"

    prefix = [f for f in os.listdir(dirname) if f.endswith("_jrc.mat")][0][:-8]
    filename = op.join(dirname, prefix + suffix)

    # `shape` as a parameter to memmap fails here
    result = np.fromfile(filename, dtype=dtype).reshape(*dims, order='F')

    return result


def load_waveforms(dirname):
    """

    Parameters
    ----------
    dirname : str
        Path to directory containing file to load.

    Returns
    -------
    result : numpy.ndarray
    """

    # standardize on num_channels x num_samples x num_events
    result = _read_jrc(dirname, "raw").swapaxes(0, 1)

    return result


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

    result = _read_matlab(dirname, "viTime_spk", flatten=True).astype(np.uint64)
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

    result = _read_matlab(dirname, "S_clu/viClu", flatten=True)

    return result.astype(np.int64) - 1

