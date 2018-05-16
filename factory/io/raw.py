"""Copyright (C) 2018 Vidrio Technologies. All rights reserved."""

import numpy as np


def read_roi(source, channels, samples):
    """

    Parameters
    ----------
    source : numpy.memmap
        Memory map of data file.
    channels : numpy.ndarray
        Channels (i.e., rows) to read.
    samples : numpy.ndarray
        Samples (i.e., columns) to read.

    Returns
    -------
    values : numpy.ndarray
        2D array of samples from data file.
    """

    assert isinstance(source, np.memmap)
    assert isinstance(channels, np.ndarray)
    assert isinstance(samples, np.ndarray)

    return source[channels[:, np.newaxis], samples[np.newaxis, :]]


def write_roi(target, channels, samples, data):
    """

    Parameters
    ----------
    target : numpy.memmap
        Memory map of data file.
    channels : numpy.ndarray
        Channels (i.e., rows) to write.
    samples : numpy.ndarray
        Samples (i.e., columns) to write.
    data : numpy.ndarray
        Data to write into target.
    """

    assert isinstance(target, np.memmap)
    assert isinstance(channels, np.ndarray)
    assert isinstance(samples, np.ndarray)
    assert isinstance(data, np.ndarray)
    assert data.shape == (channels.size, samples.size)

    target[channels[:, np.newaxis], samples[np.newaxis, :]] = data


def unit_windows(source, unit_times, samples_before, samples_after, channels=None):
    """

    Parameters
    ----------
    source : numpy.memmap
        Memory map of source data file.
    unit_times : numpy.ndarray
        Time steps around which to read samples.
    samples_before : int
        Number of samples before each unit time to read.
    samples_after : int
        Number of samples after each unit time to read.
    channels : numpy.ndarray, optional
        Channels (i.e., rows) to read from.

    Returns
    -------
    windows : numpy.ndarray
        Tensor, num_channels x num_samples x num_events.
    """

    assert isinstance(source, np.memmap)
    assert isinstance(unit_times, np.ndarray)
    assert samples_before == int(samples_before)
    assert samples_after == int(samples_after)
    assert unit_times.size > 0

    if channels is None:
        channels = np.arange(source.shape[0])
        num_channels = source.shape[0]
    else:
        assert isinstance(channels, np.ndarray)
        num_channels = channels.size

    num_samples = samples_before + samples_after + 1
    num_events = unit_times.size

    windows = np.zeros((num_channels, num_samples, num_events))

    for i in range(num_events):
        samples = np.arange(unit_times[i] - samples_before, unit_times[i] + samples_after + 1, dtype=unit_times.dtype)
        windows[:, :, i] = read_roi(source, channels, samples)

    return windows


def reset_target(source, target, samples_before, samples_after, times):
    """

    Parameters
    ----------
    source : numpy.memmap
        Memory map of source data file.
    target : numpy.memmap
        Memory map of target data file.
    samples_before : int
        Number of samples before each unit time to read.
    samples_after : int
        Number of samples after each unit time to read.
    times : numpy.ndarray
        Time steps where GT units have already been added.
    """

    assert isinstance(source, np.memmap)
    assert isinstance(target, np.memmap)
    assert source.shape == target.shape

    # it's the only way to be sure
    channels = np.arange(source.shape[0])

    for t in times:
        samples = np.arange(t - samples_before, t + samples_after + 1)
        data = read_roi(source, channels=channels, samples=samples)
        write_roi(target, channels=channels, samples=samples, data=data)
