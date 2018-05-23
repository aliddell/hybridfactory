# Copyright (C) 2018 Vidrio Technologies. All rights reserved.

import numpy as np
import pandas as pd


def build_confusion_matrix(true_labels, hybrid_labels, sort=True):
    """

    Parameters
    ----------
    true_labels : numpy.ndarray
        True labels (from curated sorting) of hybrid units.
    hybrid_labels : numpy.ndarray
        "Best match" labels from a hybrid sorting.
    sort : bool, optional
        Sort the output by best match (naively, for now).

    Returns
    -------
    confusion_matrix : pandas.DataFrame

    """

    assert isinstance(true_labels, np.ndarray)
    assert isinstance(hybrid_labels, np.ndarray)
    assert true_labels.size == hybrid_labels.size

    true_unique_labels = np.unique(true_labels)
    hybrid_unique_labels = np.unique(hybrid_labels)

    confusion_matrix = np.zeros((true_unique_labels.size, hybrid_unique_labels.size), dtype=np.int64)

    for i, true_label in enumerate(true_unique_labels):
        mask = true_labels == true_label
        hybrid_matches = hybrid_labels[mask]

        for j, hybrid_label in enumerate(hybrid_unique_labels):
            confusion_matrix[i, j] = np.count_nonzero(hybrid_matches == hybrid_label)

    if sort:
        for i in range(true_unique_labels.size):
            row = confusion_matrix[i, i:]
            sort_indices = np.flipud(np.argsort(row))

            confusion_matrix[:, i:] = confusion_matrix[:, i + sort_indices]
            hybrid_unique_labels[i:] = hybrid_unique_labels[i + sort_indices]

    # eliminate columns where all rows are zero
    keep = ~np.all(confusion_matrix == 0, axis=0)
    confusion_matrix = confusion_matrix[:, keep]
    hybrid_unique_labels = hybrid_unique_labels[keep]

    confusion_matrix =  pd.DataFrame(confusion_matrix, index=true_unique_labels, columns=hybrid_unique_labels)

    confusion_matrix.columns.name = "hybrid label"
    confusion_matrix.index.name = "true label"

    return confusion_matrix

