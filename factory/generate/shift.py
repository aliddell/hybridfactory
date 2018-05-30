# Copyright (C) 2018 Vidrio Technologies. All rights reserved.

from collections import OrderedDict

import numpy as np
import scipy.spatial

from factory.io.logging import log


def channel_coordinates(channels, probe):
    if not hasattr(channels, "__len__"):
        channels = np.array([channels])
    elif not isinstance(channels, np.ndarray):
        channels = np.array(channels)

    coords = np.zeros((channels.size, 2))
    for i, c in enumerate(channels):
        coords[i, :] = probe.channel_positions[probe.channel_map == c]

    return coords


def coordinate_channels(coords, probe):
    channels = np.zeros(coords.shape[0], dtype=probe.channel_map.dtype)

    for i, xy in enumerate(coords):
        channels[i] = probe.channel_map[np.all(probe.channel_positions == xy, axis=1)][0]

    return channels


def furthest_from(reference, subset, probe):
    if len(reference.shape) < 2:
        reference = reference[np.newaxis, :]

    subset_positions = probe.channel_positions[np.isin(probe.channel_map, subset), :]
    ai = scipy.spatial.distance.cdist(reference, subset_positions).ravel().argmax()
    anchor = probe.channel_map[np.isin(probe.channel_map, subset)][ai]

    return anchor


def find_neighbors(subset, probe):
    probe_positions = probe.channel_positions[probe.connected, :]
    probe_channels = probe.channel_map[probe.connected]
    probe_center = probe_positions.mean(axis=0)

    anchor = furthest_from(probe_center, subset, probe)
    anchor_coords = channel_coordinates(anchor, probe)
    dist_relations = channel_coordinates(subset, probe) - anchor_coords

    matches = {}
    for candidate in probe_channels:
        if candidate == anchor:
            continue
        candidate_dist_relations = probe_positions - channel_coordinates(candidate, probe)
        candidate_channels = -np.ones_like(subset)

        for i, dr in enumerate(dist_relations):
            dr_matches = np.all(candidate_dist_relations == dr, axis=1)
            if not dr_matches.any():  # no match in this relation
                break
            else:
                candidate_channels[i] = probe_channels[np.where(dr_matches)[0][0]]
        # did we survive?
        if np.count_nonzero(candidate_channels == -1) > 0:
            continue
        else:
            matches[candidate] = candidate_channels

    # now sort these in order from nearest to furthest
    match_keys = np.array(list(matches.keys()))
    match_dists = scipy.spatial.distance.cdist(anchor_coords, channel_coordinates(match_keys, probe)).ravel()

    match_order = match_dists.argsort()

    return OrderedDict([(k, matches[k]) for k in match_keys[match_order]])


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
        Channels shifted by some factor.
    """

    # inverse_channel_map[probe.channel_map] == [1, 2, ..., probe.NCHANS - 1]
    inverse_channel_map = np.zeros(probe.channel_map.size, dtype=np.int64)
    inverse_channel_map[probe.channel_map] = np.arange(probe.channel_map.size)

    if params.channel_shift is None:
        shift_candidates = find_neighbors(channels, probe)
        if not shift_candidates:
            log(f"could not find channels which replicate this spatial distribution", params.verbose)
            return None

        # bias in favor of channels further out
        weights = 1 + np.arange(len(shift_candidates), dtype=np.float64)
        weights /= weights.sum()
        anchor = np.random.choice(list(shift_candidates.keys()), p=weights)

        shifted_channels = shift_candidates[anchor]
    else:
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
