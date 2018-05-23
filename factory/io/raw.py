# Copyright (C) 2018 Vidrio Technologies. All rights reserved.

import os.path as op

import numpy as np


def open_raw(filename, data_type, num_channels, mode="r", offset=0):
    """

    Parameters
    ----------
    filename : str
        Path to file to open.
    data_type : numpy.dtype
        Type of data contained in memory-mapped file.
    num_channels : int
        Number of channels (i.e., rows) in memory-mapped file.
    mode : {'r+', 'r', 'w+', 'c'}, optional
        The file is opened in this mode:

        +------+-------------------------------------------------------------+
        | 'r'  | Open existing file for reading only.                        |
        +------+-------------------------------------------------------------+
        | 'r+' | Open existing file for reading and writing.                 |
        +------+-------------------------------------------------------------+
        | 'w+' | Create or overwrite existing file for reading and writing.  |
        +------+-------------------------------------------------------------+
        | 'c'  | Copy-on-write: assignments affect data in memory, but       |
        |      | changes are not saved to disk.  The file on disk is         |
        |      | read-only.                                                  |
        +------+-------------------------------------------------------------+

        Default is 'r'.
    offset: int, optional
        In the file, array data starts at this offset.

    Returns
    -------
    mmap : numpy.memmap
        Memory map of `filename`.
    """

    assert op.isfile(filename)
    assert num_channels == int(num_channels) and num_channels > 0
    assert mode in ("r", "r+", "w+", "c")
    assert offset == int(offset) and offset >= 0

    file_size_bytes = op.getsize(filename)
    byte_count = np.dtype(data_type).itemsize  # number of bytes in data type
    nrows = num_channels
    ncols = file_size_bytes // (nrows * byte_count)

    mmap = np.memmap(filename, dtype=data_type, offset=offset, mode=mode,
                     shape=(nrows, ncols), order="F")

    return mmap


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

    if len(channels.shape) == 1:
        channels = channels[:, np.newaxis]
    if len(samples.shape) == 1:
        samples = samples[np.newaxis, :]

    values = source[channels, samples]

    return values


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


def unit_windows(source, unit_times, samples_before, samples_after, car_channels=None):
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
    car_channels : numpy.ndarray, optional
        Channels to average and subtract.

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

    num_channels = source.shape[0]
    num_samples = samples_before + samples_after + 1
    num_events = unit_times.size

    windows = np.zeros((num_channels, num_samples, num_events))

    for i in range(num_events):
        samples = np.arange(unit_times[i] - samples_before, unit_times[i] + samples_after + 1, dtype=unit_times.dtype)
        windows[:, :, i] = read_roi(source, np.arange(num_channels), samples)

    if car_channels is not None:
        windows[car_channels] -= np.mean(windows[car_channels, :, :], axis=1)[:, np.newaxis, :]
    else:
        windows -= np.mean(windows, axis=1)[:, np.newaxis, :]

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
