# Copyright (C) 2018 Vidrio Technologies. All rights reserved.

import numpy as np
import scipy.spatial

from factory.io.logging import log


def shift_channels(channels, params, probe):
    """Shift a subset of the channels.

    Parameters
    ----------
    channels : numpy.ndarray
        Input channels to be shifted.
    params : module
        Session parameters.
    probe : module
        Probe parameters.

    Returns
    -------
    shifted_channels : numpy.ndarray or None
        Channels shifted by some constant factor.
    """

    # inverse_channel_map[probe.channel_map] == [1, 2, ..., probe.NCHANS - 1]
    inverse_channel_map = np.zeros(probe.channel_map.size, dtype=np.int64)
    inverse_channel_map[probe.channel_map] = np.arange(probe.channel_map.size)

    # make sure our shifted channels fall in the range [0, probe.channel_map)
    if inverse_channel_map[channels].max() < probe.channel_map.size - params.channel_shift:
        shifted_channels = probe.channel_map[inverse_channel_map[channels] + params.channel_shift]
    else:
        shifted_channels = probe.channel_map[inverse_channel_map[channels] - params.channel_shift]

    try:
        assert shifted_channels.min() > -1 and shifted_channels.max() < probe.channel_map.size
    except AssertionError:
        log(f"channel shift of {params.channel_shift} places events outside of probe range", params.verbose)
        return None

    # make sure our shifted channels don't land on unconnected channels
    try:
        assert np.intersect1d(shifted_channels, probe.channel_map[~probe.connected]).size == 0
    except AssertionError:
        log(f"channel shift of {params.channel_shift} places events on unconnected channels", params.verbose)
        return None

    # make sure our shifted channels don't alter spatial relationships
    channel_distance = scipy.spatial.distance.pdist(probe.channel_positions[inverse_channel_map[channels], :])
    shifted_distance = scipy.spatial.distance.pdist(probe.channel_positions[inverse_channel_map[shifted_channels], :])

    try:
        assert np.isclose(channel_distance, shifted_distance).all()
    except AssertionError:
        log(f"channel shift of {params.channel_shift} alters spatial relationship between channels", params.verbose)
        return None

    return shifted_channels


def jitter_events(unit_times, sample_rate, jitter_factor, samples_before, samples_after, upper_bound):
    """

    Parameters
    ----------
    unit_times : numpy.ndarray
        Firing times for this unit, to be jittered.

    Returns
    -------
    jittered_times : numpy.ndarray
        Jittered firing times for artificial events.
    """

    isi_samples = sample_rate // 1000  # number of samples in 1 ms
    # normally-distributed jitter factor, with an absmin of `isi_samples`
    jitter1 = isi_samples + np.abs(np.random.normal(loc=0, scale=jitter_factor // 2, size=unit_times.size // 2))
    jitter2 = -(isi_samples + np.abs(isi_samples + np.random.normal(loc=0, scale=jitter_factor // 2,
                                                                    size=unit_times.size - jitter1.size)))

    # leaves a window of 2 ms around `unit_times` so units don't fire right on top of each other
    jitter = np.random.permutation(np.hstack((jitter1, jitter2))).astype(unit_times.dtype)

    jittered_times = unit_times + jitter

    mask = (jittered_times - samples_before > 0) & (jittered_times + samples_after < upper_bound)
    jittered_times = jittered_times[mask]

    return jittered_times
