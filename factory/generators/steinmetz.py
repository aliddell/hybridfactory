import numpy as np

import factory.io.raw


def generate(source, params, probe, unit_times):
    """
    
    Parameters
    ----------
    source : numpy.memmap
    params : module
    probe : module
    unit_times : numpy.ndarray

    Returns
    -------
    recon_events : numpy.ndarray
    """
    windows = factory.io.raw.unit_windows(source, unit_times, params.samples_before, params.samples_after)
    windows[probe.channel_map[~probe.connected], :, :] = 0  # zero out the unconnected channels

    # compute mean spike and get channels by thresholding
    window_means = np.matrix(np.mean(windows, axis=2))
    window_means_shift = window_means - window_means[:, 0]

    channels = np.nonzero(np.any(window_means_shift < params.event_threshold, axis=1))[0]
    num_channels = channels.size
    num_samples = params.samples_before + params.samples_after + 1
    num_events = windows.shape[2]

    if num_channels == 0:
        return None, None

    # now create subarray for just appropriate channels
    events = windows[channels, :, :]  # num_channels x num_samples x num_events
    events_shift = events - events[:, 0, :].reshape(num_channels, 1, num_events, order="F")

    scale = np.arange(num_samples).reshape(1, num_samples, 1, order="F") / (num_samples - 1)
    events_detrended = events_shift - events_shift[:, -1, :].reshape(num_channels, 1, num_events, order="F") * scale
    events_diff = np.diff(events_detrended, axis=1)

    # compute the SVD on the derivative of the waveforms
    flat_spikes = events_diff.reshape(num_channels * (num_samples - 1), num_events, order="F")
    u, s, vt = np.linalg.svd(flat_spikes, full_matrices=True)

    # take just the most significant singular values
    u = np.matrix(u[:, :params.num_singular_values])
    s = np.matrix(np.diag(s[:params.num_singular_values]))
    vt = np.matrix(vt[:params.num_singular_values, :])

    # reconstruct artificial event derivatives from SVD and integrate
    flat_recon_events = np.array(u * s * vt)
    recon_events = np.hstack(
        (np.zeros((num_channels, 1, num_events)),
         np.cumsum(flat_recon_events.reshape(num_channels, num_samples - 1, num_events, order="F"), axis=1))
    )

    return recon_events, channels
