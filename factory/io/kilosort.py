# Copyright (C) 2018 Vidrio Technologies. All rights reserved.

import os.path as op

import h5py
import numpy as np
import scipy.io


def _read_matlab(dirname, field, flatten=False):
    """

    Parameters
    ----------
    dirname : str
        Path to directory containing rez.mat.
    field : str
        Field to read from MAT file.

    Returns
    -------
    result : numpy.ndarray
    """

    filename = op.join(dirname, "rez.mat")
    assert op.isfile(filename)

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

    st3 = _read_matlab(dirname, "rez/st3")
    return st3[:, 1].astype(np.uint32) - 1


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

    # adapted from rezToPhy.m
    U = _read_matlab(dirname, "rez/U")
    W = _read_matlab(dirname, "rez/W")

    nt0 = W.shape[0]
    n_filt = _read_matlab(dirname, "rez/ops/Nfilt", flatten=True)[0].astype(np.int64)
    n_chan = _read_matlab(dirname, "rez/ops/Nchan", flatten=True)[0].astype(np.int64)

    templates = np.zeros((n_filt, nt0, n_chan), dtype=np.float32)

    for i in range(n_filt):
        templates[i, :, :] = (np.matrix(U[:, i, :]) * np.matrix(W[:, i, :]).T).T

    return templates


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

    st3 = _read_matlab(dirname, "rez/st3")
    result = st3[:, 0].astype(np.int64) - 1

    assert (result == np.sort(result)).all()
    assert result[0] >= 0

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

    st3 = _read_matlab(dirname, "rez/st3")
    if st3.shape[1] > 4:
        result = st3[:, 4]
    else:
        result = st3[:, 1]

    return result.astype(np.int64) - 1

