import os.path as op

import numpy as np


def save_gt_units(dirname, gt_channels, gt_times, gt_labels):
    """

    Parameters
    ----------
    dirname : str
    gt_channels :
    gt_times :
    gt_labels :

    Returns
    -------

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

    np.save(op.join(dirname, "firings_true.npy"), firings_true)
