import numpy as np

SPIKE_LIMIT = 25000


def read_roi(source, channels, samples):
    """

    Parameters
    ----------
    source : numpy.memmap
        Memory map of data file.
    channels : numpy.ndarray
    samples : numpy.ndarray

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
    samples : numpy.ndarray
    data : numpy.ndarray

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
    unit_times : numpy.ndarray
    samples_before : int
    samples_after : int
    channels : numpy.ndarray, optional

    Returns
    -------

    """
    if channels is None:
        channels = np.arange(source.shape[0])
        num_channels = source.shape[0]
    num_samples = samples_before + samples_after + 1
    num_events = unit_times.size

    if num_events > SPIKE_LIMIT:
        unit_times = np.random.choice(unit_times, size=SPIKE_LIMIT, replace=False)
        num_events = SPIKE_LIMIT

    windows = np.zeros((num_channels, num_samples, num_events))

    for i in range(num_events):
        samples = np.arange(unit_times[i] - samples_before, unit_times[i] + samples_after + 1)
        windows[:, :, i] = read_roi(source, channels, samples)

    return windows
