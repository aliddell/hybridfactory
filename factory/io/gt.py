# Copyright (C) 2018 Vidrio Technologies. All rights reserved.

import os.path as op

import numpy as np


def save_gt_units(dirname, gt_channels, gt_times, gt_labels):
    """

    Parameters
    ----------
    dirname : str
        Path to directory in which to save units.
    gt_channels : iterable
        List of numpy.ndarray, central channels on which events occur.
    gt_times : iterable
        List of numpy.ndarray, sample times at which events occur.
    gt_labels : iterable
        List of int, unique cluster labels of units.

    Returns
    -------
    filename : str
        Path to file in which data is saved.
    """

    counts = [c.size for c in gt_channels]
    assert counts == [t.size for t in gt_times]
    assert len(counts) == len(gt_labels)

    # join all channels, times, labels into single arrays
    channels = np.hstack(gt_channels)
    times = np.hstack(gt_times)
    labels = np.repeat(gt_labels, counts)

    # sort all by sample time, ascending
    ordering = times.argsort()

    firings_true = np.zeros((3, np.sum(counts)), dtype=np.uint64)

    firings_true[0, :] = channels[ordering]
    firings_true[1, :] = times[ordering]
    firings_true[2, :] = labels[ordering]

    filename = op.join(dirname, "firings_true.npy")

    np.save(filename, firings_true)

    return filename


def load_gt_units(dirname):
    """

    Parameters
    ----------
    dirname : str
        Path to directory containing file to load.

    Returns
    -------
    firings_true : numpy.ndarray
        Matrix of firing times, channels, and labels for ground-truth units.
    """

    filename = op.join(dirname, "firings_true.npy")
    assert op.isfile(filename)

    return np.load(filename)
