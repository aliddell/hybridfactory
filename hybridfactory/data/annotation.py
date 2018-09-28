# Copyright (C) 2018 Vidrio Technologies. All rights reserved.

from collections import OrderedDict
import os
from os import path as op

import h5py
import numpy as np
import pandas as pd
import scipy.io


def _read_npy(filename, flatten=False):
    """Read from a NumPy .npy file.

    Parameters
    ----------
    filename : str
        Path to file to load.

    Returns
    -------
    result : numpy.ndarray
    """

    assert op.isfile(filename)

    result = np.load(filename)

    if flatten:
        result = result.ravel()

    return result


def _read_matlab(filename, field, flatten=False):
    """Read from a Matlab .mat file.

    Parameters
    ----------
    filename : str
        Path to .mat file to read.
    field : str
        Field to read from MAT file.

    Returns
    -------
    result : numpy.ndarray
    """

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


def _read_jrc(filename, dims, dtype):
    """Read from a JRCLUST .jrc file.

    Parameters
    ----------
    filename : str
        Path to file containing data.
    dims : iterable
        Dimensions of data in file.
    dtype : type
        NumPy data type of data in file.

    Returns
    -------
    result : numpy.ndarray
    """

    assert op.isfile(filename)
    assert isinstance(dtype, type)
    assert len(dims) == 3

    # `shape` as a parameter to memmap fails here
    result = np.fromfile(filename, dtype=dtype).reshape(*dims, order='F')

    return result


def _jrc_prefix(dirname):
    """Get a JRCLUST session prefix.

    Parameters
    ----------
    dirname : str
        Path to directory containing JRCLUST output.

    Returns
    -------
    prefix : str
        Session name for JRCLUST sorting.

    Raises
    ------
    ValueError
        If more than one distinct session is found in directory.

    """

    assert op.isdir(dirname)
    ls = os.listdir(dirname)

    matfiles = [f for f in ls if f.endswith("_jrc.mat")]
    assert len(matfiles) > 0

    if len(matfiles) > 1:  # more than one file ending in _jrc.mat -- which one do we take?
        raise ValueError(f"ambiguous data directory: {dirname}")

    return matfiles[0][:-8]


def kilosort_from_rez(dirname, rezfile="rez.mat"):
    """Load a KiloSort sorting from rez.mat.

    Parameters
    ----------
    dirname : str
        Path to directory containing KiloSort output (rez.mat).

    Returns
    -------
    event_annotations : pandas.DataFrame
        Zero-based timesteps, one-based cluster IDs, zero-based template IDs.
    """

    assert op.isdir(dirname)
    assert rezfile in os.listdir(dirname)

    filename = op.join(dirname, rezfile)
    st3 = _read_matlab(filename, "rez/st3")

    # times and template IDs are zero-based since they serve as indices into data
    event_times = st3[:, 0].astype(np.int64) - 1
    event_templates = st3[:, 1].astype(np.int64) - 1

    # cluster IDs are not indices, so can be one-based
    if st3.shape[1] > 4:
        event_clusters = st3[:, 4].astype(np.int64)
    else:
        event_clusters = event_templates + 1

    # sanity check
    assert (event_times == np.sort(event_times)).all()
    assert event_times[0] >= 0
    assert (event_templates >= 0).all()
    assert (event_clusters >= 0).all()
    assert event_times.shape == event_templates.shape == event_clusters.shape

    return pd.DataFrame(data=OrderedDict([("timestep", event_times),
                                          ("cluster", event_clusters),
                                          ("template", event_templates)]))


def kilosort_from_phy(dirname):
    """Load a Phy-compatible sorting from `*.npy` files.

    Parameters
    ----------
    dirname : str
        Path to directory containing Phy-compatible input (`*.npy`).

    Returns
    -------
    event_annotations : pandas.DataFrame
        Zero-based timesteps, one-based cluster IDs, zero-based template IDs.
    """

    assert op.isdir(dirname)
    ls = os.listdir(dirname)
    for filename in ("spike_times.npy", "spike_templates.npy", "spike_clusters.npy"):
        assert filename in ls

    # times and template IDs are zero-based since they serve as indices into data
    event_times = _read_npy(op.join(dirname, "spike_times.npy"), flatten=True).astype(np.int64) - 1
    event_templates = _read_npy(op.join(dirname, "spike_templates.npy"), flatten=True).astype(np.int64)

    # cluster IDs are not indices, so can be one-based
    event_clusters = _read_npy(op.join(dirname, "spike_clusters.npy"), flatten=True).astype(np.int64) + 1

    # sanity check
    assert (event_times == np.sort(event_times)).all()
    assert event_times[0] >= 0
    assert (event_templates >= 0).all()
    assert (event_clusters >= 0).all()
    assert event_times.shape == event_templates.shape == event_clusters.shape

    return pd.DataFrame(data=OrderedDict([("timestep", event_times),
                                          ("cluster", event_clusters),
                                          ("template", event_templates)]))


def jrclust_from_matfile(dirname, matfile=None, consolidate=True):
    """Load a JRCLUST sorting.

    Parameters
    ----------
    dirname : str
        Path to directory containing JRCLUST output (`*_jrc.mat`, `*.jrc`).
    consolidate : bool
        Consolidate all negative cluster IDs into -1 if True.

    Returns
    -------
    event_annotations : pandas.DataFrame
        Zero-based timesteps, integer cluster IDs, zero-based channel indices.
    """

    if matfile is None:
        prefix = _jrc_prefix(dirname)  # handles assertions for us
    else:
        prefix = op.basename(matfile).replace("_jrc.mat", "")

    filename = op.join(dirname, f"{prefix}_jrc.mat")

    # times and channel indices are zero-based since they serve as indices into data
    event_times = _read_matlab(filename, "viTime_spk", flatten=True).astype(np.int64) - 1
    event_channel_indices = _read_matlab(filename, "viSite_spk", flatten=True).astype(np.int64) - 1

    # cluster IDs are not indices (and can in fact be negative)
    event_clusters = _read_matlab(filename, "S_clu/viClu", flatten=True)

    # sanity check
    assert (event_times == np.sort(event_times)).all()
    assert event_times[0] >= 0
    assert event_times.shape == event_clusters.shape

    ea = pd.DataFrame(data=OrderedDict([("timestep", event_times),
                                        ("cluster", event_clusters),
                                        ("channel_index", event_channel_indices)]))

    # consolidate garbage clusters
    if consolidate:
        ea.loc[ea[ea.cluster < 0].index, "cluster"] = -1

    return ea


def load_kilosort_templates(dirname):
    """Load templates output by KiloSort.

    Parameters
    ----------
    dirname : str
        Path to directory containing either KiloSort output or at least Phy input.

    Returns
    -------
    templates : numpy.ndarray
        KiloSort templates as a 3-dimensional array.

    Raises
    ------
    ValueError
        If `dirname` does not contain either templates.npy or rez.mat.
    """

    assert op.isdir(dirname)
    npy_filename = op.join(dirname, "templates.npy")
    ks_filename = op.join(dirname, "rez.mat")

    if op.isfile(npy_filename):  # this is the easy option
        templates = _read_npy(npy_filename)
    elif op.isfile(ks_filename):  # adapted from rezToPhy
        U = _read_matlab(ks_filename, "rez/U")
        W = _read_matlab(ks_filename, "rez/W")

        nt0 = W.shape[0]
        n_filt = _read_matlab(ks_filename, "rez/ops/Nfilt", flatten=True)[0].astype(np.int64)
        n_chan = _read_matlab(ks_filename, "rez/ops/Nchan", flatten=True)[0].astype(np.int64)

        templates = np.zeros((n_filt, nt0, n_chan), dtype=np.float32)

        for i in range(n_filt):
            templates[i, :, :] = (np.matrix(U[:, i, :]) * np.matrix(W[:, i, :]).T).T
    else:  # nothing to load!
        raise ValueError(f"no obvious source for templates: {dirname}")

    return templates


def load_jrc_raw(dirname):
    """Load raw waveforms output by JRCLUST.

    Parameters
    ----------
    dirname : str
        Path to directory containing JRCLUST output (`*_jrc.mat`, `*.jrc`).

    Returns
    -------
    result : numpy.ndarray
        Raw event waveforms.
    """

    prefix = _jrc_prefix(dirname)  # handles assertions for us
    filename = op.join(dirname, f"{prefix}_spkraw.jrc")
    dims = tuple(_read_matlab(op.join(dirname, f"{prefix}_jrc.mat"), "dimm_raw", flatten=True).astype(np.int32))
    dtype = np.int16

    return _read_jrc(filename, dims, dtype)


def load_jrc_filtered(dirname):
    """Load filtered waveforms output by JRCLUST.

    Parameters
    ----------
    dirname : str
        Path to directory containing JRCLUST output (`*_jrc.mat`, `*.jrc`).

    Returns
    -------
    result : numpy.ndarray
        Filtered event waveforms.
    """

    prefix = _jrc_prefix(dirname)  # handles assertions for us
    filename = op.join(dirname, f"{prefix}_spkwav.jrc")
    dims = tuple(_read_matlab(op.join(dirname, f"{prefix}_jrc.mat"), "dimm_spk", flatten=True).astype(np.int32))
    dtype = np.int16

    return _read_jrc(filename, dims, dtype)


def load_jrc_features(dirname):
    """Load features output by JRCLUST.

    Parameters
    ----------
    dirname : str
        Path to directory containing JRCLUST output (`*_jrc.mat`, `*.jrc`).

    Returns
    -------
    result : numpy.ndarray
        Event features.
    """

    prefix = _jrc_prefix(dirname)  # handles assertions for us
    filename = op.join(dirname, f"{prefix}_spkfet.jrc")
    dims = tuple(_read_matlab(op.join(dirname, f"{prefix}_jrc.mat"), "dimm_fet", flatten=True).astype(np.int32))
    dtype = np.float32

    return _read_jrc(filename, dims, dtype)


def load_ground_truth_matrix(dirname):
    """Load ground-truth matrix stored in firings_true.npy

    Parameters
    ----------
    dirname : str
        Path to directory containing firings_true.npy.

    Returns
    -------
    result : numpy.ndarray
        Ground-truth matrix.
    """

    if not op.isdir(dirname):
        raise ValueError(f"not a directory: {dirname}")
    if not "firings_true.npy" in os.listdir(dirname):
        raise IOError(f"firings_true not in directory: {dirname}")

    filename = op.join(dirname, "firings_true.npy")

    result = _read_npy(filename).astype(np.int64)

    return result
