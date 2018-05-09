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

    channels = np.hstack(gt_channels)
    times = np.hstack(gt_times)
    labels = np.repeat(gt_labels, counts)

    firings_true = np.zeros((3, np.sum(counts)), dtype=np.uint64)

    firings_true[0, :] = channels
    firings_true[1, :] = times
    firings_true[2, :] = labels

    np.save(op.join(dirname, "firings_true.npy"), firings_true)
