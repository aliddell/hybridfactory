"""Copyright (C) 2018 Vidrio Technologies. All rights reserved."""

import numpy as np

import factory.io


def steinmetz(events, num_singular_values):
    """

    Parameters
    ----------
    events : numpy.ndarray
        Raw waveforms of events identified with this unit.
    num_singular_values : int
        Number of singular values to use in reconstruction of events.

    Returns
    -------
    recon_events : numpy.ndarray
        Tensor, num_channels x num_samples x num_events.
    """

    assert isinstance(events, np.ndarray)
    assert len(events.shape) == 3
    assert np.prod(events.shape) > 0

    assert 0 < num_singular_values <= min(np.prod(events.shape[:2]), events.shape[2])
    assert num_singular_values == int(num_singular_values)

    num_channels, num_samples, num_events = events.shape

    events_shift = events - events[:, 0, :].reshape(num_channels, 1, num_events, order="F")

    scale = np.arange(num_samples).reshape(1, num_samples, 1, order="F") / (num_samples - 1)
    events_detrended = events_shift - events_shift[:, -1, :].reshape(num_channels, 1, num_events, order="F") * scale
    events_diff = np.diff(events_detrended, axis=1)

    # compute the SVD on the derivative of the waveforms
    flat_spikes = events_diff.reshape(num_channels * (num_samples - 1), num_events, order="F")
    u, s, vt = np.linalg.svd(flat_spikes, full_matrices=True)

    # take just the most significant singular values
    u = np.matrix(u[:, :num_singular_values])
    s = np.matrix(np.diag(s[:num_singular_values]))
    vt = np.matrix(vt[:num_singular_values, :])

    # reconstruct artificial event derivatives from SVD and integrate
    flat_recon_events = np.array(u * s * vt)
    recon_events = np.hstack(
        (np.zeros((num_channels, 1, num_events)),
         np.cumsum(flat_recon_events.reshape(num_channels, num_samples - 1, num_events, order="F"), axis=1))
    )

    return recon_events
