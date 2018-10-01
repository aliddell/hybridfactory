# Copyright (C) 2018 Vidrio Technologies. All rights reserved.

from collections import OrderedDict
import datetime
import glob
import numbers
import os
from os import path as op
import shutil

import numpy as np
import pandas as pd

import hybridfactory.data.annotation
from hybridfactory.probes.probe import Probe, save_probe, load_probe
from hybridfactory.io.spikegl import get_start_times


class DataSet(object):
    def __init__(self, filename, dtype, sample_rate, probe, start_times):
        """

        Parameters
        ----------
        filename : str
            Path to file containing raw data, or glob of such files.
        dtype : type
            NumPy data type of raw data contained in `filename`.
        sample_rate : int
            Sample rate in Hz of raw data collected.
        probe : Probe
            Object representing probe with which this data was collected.
        start_times : iterable
            In the case of multiple data files, offsets for each file in absolute sample time.
        """

        if not isinstance(dtype, type):
            raise TypeError("dtype must be a NumPy dtype")

        self._dtype = dtype

        if not isinstance(probe, Probe):
            raise TypeError("probe must be an instance of Probe")

        self._probe = probe

        if not isinstance(sample_rate, numbers.Integral):
            raise TypeError("sample_rate must be an integer")
        elif sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        self._sample_rate = sample_rate

        if isinstance(filename, str):
            filenames = glob.glob(op.abspath(filename))
        elif hasattr(filename, "__iter__"):
            assert all([op.isfile(f) for f in filename])
            filenames = filename
        else:
            filenames = None

        if not filenames:
            raise IOError(f"no such file or bad glob: {filename}")

        if not hasattr(start_times, "__iter__"):
            start_times = np.array([start_times])
        else:
            start_times = np.array(start_times)

        if start_times.size != len(filenames):
            raise ValueError(f"count of start times ({start_times.size}) does not match number of files ({len(files)})")

        if not np.issubdtype(start_times.dtype, np.integer):
            raise TypeError("start_times must be integers")
        elif not (start_times >= 0).all():
            raise ValueError("start_times must be nonnegative")

        n_samples = [op.getsize(f) // (np.dtype(dtype).itemsize * probe.num_channels) for f in filenames]

        # filewise metadata
        self._metadata = pd.DataFrame(OrderedDict([("filename", filenames),
                                                   ("samples", n_samples),
                                                   ("start_time", start_times)]))
        # sort by time and reindex
        self._metadata.sort_values("start_time", inplace=True)
        self._metadata.index = np.arange(self._metadata.shape[0])

        self._data = None
        self._mode = None

    @property
    def dtype(self):
        return self._dtype

    @property
    def filenames(self):
        return self._metadata.filename.values.tolist()

    @property
    def isopen(self):
        return self._data is not None

    @property
    def metadata(self):
        return self._metadata

    @property
    def mode(self):
        return self._mode

    @property
    def probe(self):
        return self._probe

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def start_times(self):
        return self._metadata.start_time.values

    def close_raw(self):
        """Close raw data file(s)."""
        if self._data is not None:
            for mmap in self._data.items():
                del mmap
            self._data = None
            self._mode = None

    def export_metadata(self, filename):
        """Save annotations to file.

        Parameters
        ----------
        filename : str
            Path to CSV file in which to save metadata.
        """

        self._metadata.to_csv(filename, index=False)

    def last_sample(self):
        last_i = self._metadata.last_valid_index()
        return self._metadata.loc[last_i, "start_time"] + self._metadata.loc[last_i, "samples"]

    def open_raw(self, mode="r"):
        """Open raw data file(s).

        Parameters
        ----------
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
        """

        assert mode in ("r", "r+", "w+", "c")

        if self._data is None:
            self._data = {}
            for filename, n_samples, _ in self._metadata.itertuples(index=False, name=None):
                self._data[filename] = np.memmap(filename, dtype=self._dtype, shape=(self._probe.num_channels, n_samples),
                                                 mode=mode, offset=0, order='F')
            self._mode = mode

    def read_roi(self, channels, samples):
        """Read a region from the underlying raw data file(s).

        Parameters
        ----------
        channels : numpy.ndarray
            Channels (i.e., rows) to read.
        samples : numpy.ndarray
            Samples (i.e., columns) to read.

        Returns
        -------
        values : numpy.ndarray
            2D array of samples from data file.

        """

        if self._data is None:
            raise UnopenedDataException("this data set is not open for reading")
        else:
            assert isinstance(channels, np.ndarray)
            assert channels.min() >= 0 and channels.max() < self._probe.num_channels
            assert isinstance(samples, np.ndarray)
            assert samples.min() >= 0 and samples.max() < self.last_sample()

            if len(channels.shape) == 1:
                channels = channels[:, np.newaxis]

            file_indices = np.searchsorted(self._metadata.start_time, samples, side="right") - 1

            if (file_indices < 0).any():
                raise ValueError("sample indices outside of data bounds")

            values = np.zeros((channels.size, samples.size), dtype=self._dtype)

            for i in np.unique(file_indices):
                mask = file_indices == i
                lower_bound = self._metadata.loc[i, "start_time"]
                upper_bound = lower_bound + self._metadata.loc[i, "samples"]

                subsamples = samples[mask][np.newaxis, :]
                if not ((subsamples >= lower_bound).all() and (subsamples < upper_bound).all()):
                    raise ValueError("sample indices outside of data bounds")

                data = self._data[self._metadata.loc[i, "filename"]][channels, subsamples - lower_bound]
                values[:, mask] = data

            return values

    def write_roi(self, channels, samples, values):
        """Write a region to the underlying raw data file(s).

        Parameters
        ----------
        channels : numpy.ndarray
            Channels (i.e., rows) to write.
        samples : numpy.ndarray
            Samples (i.e., columns) to write.
        values : numpy.ndarray
            Data to write into target.
        """

        if self._data is None or self._mode == 'r':
            raise UnopenedDataException("this data set is not open for writing")
        else:
            assert isinstance(channels, np.ndarray)
            assert channels.min() >= 0 and channels.max() < self._probe.num_channels
            assert isinstance(samples, np.ndarray)
            assert samples.min() >= 0 and samples.max() < self.last_sample()
            assert isinstance(values, np.ndarray)
            assert values.shape == (channels.size, samples.size)

            if len(channels.shape) == 1:
                channels = channels[:, np.newaxis]

            file_indices = np.searchsorted(self._metadata.start_time, samples, side="right") - 1

            if (file_indices < 0).any():
                raise ValueError("sample indices outside of data bounds")

            for i in np.unique(file_indices):
                mask = file_indices == i
                lower_bound = self._metadata.loc[i, "start_time"]
                upper_bound = lower_bound + self._metadata.loc[i, "samples"]

                subsamples = samples[mask][np.newaxis, :]
                if not ((subsamples >= lower_bound).all() and (subsamples < upper_bound).all()):
                    raise ValueError("sample indices outside of data bounds")

                data = self._data[self._metadata.loc[i, "filename"]]
                data[channels, (subsamples - lower_bound)[np.newaxis, :]] = values[:, mask]


class AnnotatedDataSet(DataSet):
    def __init__(self, filename, dtype, sample_rate, probe, start_times, annotations):
        """

        Parameters
        ----------
        filename : str
            Path to file containing raw data, or glob of such files.
        dtype : type
            NumPy data type of raw data contained in `filename`.
        sample_rate : int
            Sample rate in Hz of raw data collected.
        probe : Probe
            Object representing probe with which this data was collected.
        start_times : iterable
            In the case of multiple data files, offsets for each file in absolute sample time.
        annotations : pandas.DataFrame
            Timestep, cluster, and algorithm-specific metadata of clustering.
        """
        super().__init__(filename, dtype, sample_rate, probe, start_times)

        if not isinstance(annotations, pd.DataFrame):
            raise TypeError("annotations must be a pandas DataFrame")

        assert "timestep" in annotations.columns and "cluster" in annotations.columns

        self._annotations = annotations

    @property
    def annotations(self):
        return self._annotations

    def export_annotations(self, filename):
        """Save annotations to file.

        Parameters
        ----------
        filename : str
            Path to CSV file in which to save annotations.
        """

        self._annotations.to_csv(filename, index=False)

    def unit_channels(self, unit, threshold=0, samples_before=40, samples_after=40):
        """Find relevant channels in `unit`.

        Parameters
        ----------
        unit : int
            Cluster ID of unit of interest.
        threshold : int
            Negative threshold which a channel must surpass to be considered relevant.
        samples_before : int
            Number of samples before each unit time to read.
        samples_after : int
            Number of samples after each unit time to read.

        Returns
        -------
        channels : numpy.ndarray
        """

        if not isinstance(threshold, numbers.Integral):
            raise TypeError("threshold must be an integer")

        if "channel_index" in self.annotations.columns:
            channels = self.annotations.loc[self.annotations.cluster == unit, "channel_index"].values
        else:
            if not self.isopen:
                self.open_raw()
                close_self = True
            else:
                close_self = False

            if threshold >= 0:
                raise ValueError("threshold must be negative")

            windows = self.unit_windows(unit, samples_before, samples_after, car=True)

            # no events found!
            if windows.shape[2] == 0:
                return np.array([])

            # compute mean spike and get channels by thresholding
            window_means = np.mean(windows, axis=2)
            window_means_shift = window_means - window_means[:, 0][:, np.newaxis]

            channels = np.nonzero(np.any(window_means_shift < threshold, axis=1))[0]

            if close_self:
                self.close_raw()

        return channels

    def unit_event_count(self, unit):
        try:
            count = self.annotations[self.annotations.cluster == unit].shape[0]
        except TypeError:
            count = 0

        return count

    def unit_firing_times(self, unit):
        try:
            times = self.annotations[self.annotations.cluster == unit].timestep.values
        except TypeError:
            times = np.array([])

        return times

    def unit_windows(self, unit, samples_before, samples_after, car=True):
        """Find windows around events in cluster `unit`.

        Parameters
        ----------
        unit : int
            Cluster ID of unit of interest.
        samples_before : int
            Number of samples before each unit time to read.
        samples_after : int
            Number of samples after each unit time to read.
        car : bool, optional
            Perform common-average-referencing on connected channels.

        Returns
        -------
        windows : numpy.ndarray
            Tensor, num_channels x num_samples x num_events.
        """

        if unit not in self.annotations.cluster:
            raise ValueError(f"can't find unit {unit}")

        if not isinstance(samples_before, numbers.Integral):
            raise TypeError("samples_before must be an integer")
        elif samples_before <= 0:
            raise ValueError("samples_before must be positive")

        if not isinstance(samples_after, numbers.Integral):
            raise TypeError("samples_after must be an integer")
        elif samples_after <= 0:
            raise ValueError("samples_after must be positive")

        unit_times = self._annotations[self._annotations.cluster == unit].timestep.values

        num_channels = self.probe.num_channels
        num_samples = samples_before + samples_after + 1
        num_events = unit_times.size

        windows = np.zeros((num_channels, num_samples, num_events))

        for i in range(num_events):
            samples = np.arange(unit_times[i] - samples_before, unit_times[i] + samples_after + 1,
                                dtype=unit_times.dtype)
            windows[:, :, i] = self.read_roi(np.arange(num_channels), samples)

        if car:
            car_channels = self.probe.channel_map[self.probe.connected]
            windows[car_channels] -= np.mean(windows[car_channels, :, :], axis=1)[:, np.newaxis, :]

        return windows[:, :]


class HybridDataSet(AnnotatedDataSet):
    """

    """
    def __init__(self, source, filename):
        """

        Parameters
        ----------
        source : AnnotatedDataSet
            Starting point for hybrid data set.
        filename : str
            Path to file containing raw data, or glob of such files.
        """
        assert isinstance(source, AnnotatedDataSet)
        super().__init__(filename, source.dtype, source.sample_rate, source.probe, source.start_times.copy(),
                         source.annotations.copy())

        self._artificial_units = pd.DataFrame(OrderedDict([("timestep", []),
                                                           ("true_unit", []),
                                                           ("new_unit", []),
                                                           ("center_channel", [])]), dtype=np.int64)

    @property
    def artificial_units(self):
        return self._artificial_units

    def export_artificial_units(self, filename):
        """Save artificial units to CSV file.

        Parameters
        ----------
        filename : str
            Path to CSV file in which to save artificial units.
        """

        self._artificial_units.to_csv(filename, index=False)

    def export_ground_truth_matrix(self, filename):
        """Save artificial units to a NumPy binary file.

        Parameters
        ----------
        filename : str
            Path to file to write.
        """
        au = self.artificial_units.sort_values("timestep")
        gt_matrix = au.values.T[[3, 0, 1]].astype(np.uint64)  # new order is channel, timestep, unit

        np.save(filename, gt_matrix)

    def reset(self, source):
        """Reset hybrid data set to a copy of `source`.

        Parameters
        ----------
        source : AnnotatedDataSet
        """

        # do some sanity checks
        if not isinstance(source, AnnotatedDataSet):
            raise TypeError("source must be an AnnotatedDataSet")
        if source.probe != self.probe:
            raise ValueError("source and self do not share the same probe")
        if source.start_times != self.start_times:
            raise ValueError("source and self do not have the same start times")
        if not all([op.getsize(fn) == op.getsize(self.filenames[k]) for k, fn in enumerate(source.filenames)]):
            raise ValueError("source and self do not have the same file sizes")

        if not self.isopen:
            self.open_raw("r+")
            close_self = True
        else:
            close_self = False

        if not source.isopen:
            close_source = True
            source.open_raw("r")
        else:
            close_source = False

        # it's the only way to be sure
        channels = np.arange(self.probe.num_channels)

        end_t = self.last_sample()

        for t in self._artificial_units.timestep:
            # find beginning of window
            start_t = t - 1
            source_col = source.read_roi(channels, np.array([start_t]))
            hybrid_col = self.read_roi(channels, np.array([start_t]))
            while not np.isclose(source_col, hybrid_col).all() and start_t >= 0:
                start_t -= 1
                source_col = source.read_roi(channels, np.array([start_t]))
                hybrid_col = self.read_roi(channels, np.array([start_t]))

            if start_t == -1:
                raise HybridSourceMismatchError("bad interval")

            # find end of window
            stop_t = t + 1
            source_col = source.read_roi(channels, np.array([stop_t]))
            hybrid_col = self.read_roi(channels, np.array([stop_t]))
            while not np.isclose(source_col, hybrid_col).all() and stop_t < end_t:
                stop_t += 1
                source_col = source.read_roi(channels, np.array([stop_t]))
                hybrid_col = self.read_roi(channels, np.array([stop_t]))

            if stop_t == end_t:
                raise HybridSourceMismatchError("bad interval")

            # if we're successful in finding where the window starts and stops, overwrite
            samples = np.arange(start_t, stop_t)
            source_roi = source.read_roi(channels, samples)
            self.write_roi(channels, samples, source_roi)

        if close_self:
            self.close_raw()
        if close_source:
            source.close_raw()

        # reset annotations and artificial units
        self._annotations = source.annotations.copy()
        self._artificial_units.drop(self._artificial_units.index, inplace=True)


def new_annotated_dataset(filename, dtype, sample_rate, probe, ann_location=None, ann_format=None):
    """Create a new annotated dataset from spike sorter output.

    Parameters
    ----------
    filename : str
        Path to file containing raw data, or glob of such files.
    dtype : type
        NumPy type of raw data.
    sample_rate : int
        Sample rate in Hz of raw data collected.
    probe : Probe
        Probe object.
    ann_location : str, optional
        Path to directory containing annotations.
    ann_format : {"phy", "kilosort", "jrc"}, optional
        Annotation format.

    Returns
    -------
    AnnotatedDataSet
    """

    if not glob.glob(filename):
        raise ValueError(f"no such file or bad glob: {filename}")
    if ann_format is not None and ann_format not in ("phy", "kilosort", "jrc"):
        raise ValueError("ann_format must be one of 'phy', 'kilosort', or 'jrc'")

    if not ann_location:  # try to load from the same directory
        ann_location = op.dirname(filename)

    assert op.isdir(ann_location)

    try:
        last_dot = -(filename[::-1].index('.') + 1)
        meta_filename = filename[:last_dot] + ".meta"
        start_times = get_start_times(meta_filename)
    except IOError:  # no '.' found in filename, no firstSample, no such file, or bad glob
        start_times = [0]

    if ann_format == "phy":
        annotations = hybridfactory.data.annotation.kilosort_from_phy(ann_location)
    elif ann_format == "kilosort":
        annotations = hybridfactory.data.annotation.kilosort_from_rez(ann_location)
    elif ann_format == "jrc":
        annotations = hybridfactory.data.annotation.jrclust_from_matfile(ann_location)
    else:  # try to infer the annotation format
        ls = os.listdir(ann_location)
        if "spike_times.npy" in ls and "spike_templates.npy" in ls and "spike_clusters.npy" in ls:
            annotations = hybridfactory.data.annotation.kilosort_from_phy(ann_location)
        elif "rez.mat" in ls:
            annotations = hybridfactory.data.annotation.kilosort_from_rez(ann_location)
        elif any([f.endswith("_jrc.mat") for f in ls]):
            annotations = hybridfactory.data.annotation.jrclust_from_matfile(ann_location)
        else:
            raise ValueError("cannot locate any annotations")

    return AnnotatedDataSet(filename, dtype, sample_rate, probe, start_times, annotations)


def new_hybrid_dataset(source, output_directory, copy=True, create=True, transform=None):
    """Create a new hybrid dataset from an annotated dataset.

    Parameters
    ----------
    source : AnnotatedDataSet
    output_directory : str
        Path to directory to contain hybrid data.
    copy : bool, optional
        Copy source data to output data.
    create : bool, optional
        Create `output_directory` if it doesn't already exist.
    transform : function, optional
        Takes a string and returns a string.

    Returns
    -------
    HybridDataSet
    """

    def _transform(rtf):
        try:
            last_dot = -(rtf[::-1].index('.') + 1)
            rtf = rtf[:last_dot] + ".GT" + rtf[last_dot:]  # add ".GT" before extension
        except ValueError:  # no '.' found in rtf
            rtf += ".GT"  # add ".GT" at the end

        return rtf

    if not isinstance(source, AnnotatedDataSet):
        raise TypeError("source must be an instance of AnnotatedDataSet")

    if transform is None:
        transform = _transform

    if not op.isdir(output_directory):
        if create:
            os.mkdir(output_directory)
        else:
            raise IOError("output_directory does not exist but you have requested not to create it")

    old_filenames = source.filenames
    new_filenames = old_filenames.copy()

    for k, filename in enumerate(new_filenames):
        src_dir = op.dirname(filename)
        new_filenames[k] = transform(filename).replace(src_dir, output_directory)

    if copy:
        for k, ofn in enumerate(old_filenames):
            shutil.copy2(ofn, new_filenames[k])

    return HybridDataSet(source, new_filenames)


def load_dataset(dirname, **kwargs):
    """Hackily load a DataSet's metadata from a set of text and binary files and return the DataSet.

    Parameters
    ----------
    dirname : str
        Path to directory containing dataset save files.
    metadata_file : str, optional
    annotations_file : str, optional
    artificial_units_file : str, optional
    probe_file : str, optional
    dtype_file : str, optional

    Returns
    -------
    DataSet
    """

    if not op.isdir(dirname):
        raise ValueError(f"{dirname} is not a directory")

    ls = os.listdir(dirname)

    # check for files if in kwargs
    if "metadata_file" in kwargs:
        metadata_file = kwargs["metadata_file"]
        if metadata_file != op.abspath(metadata_file):
            metadata_file = op.join(dirname, metadata_file)
    else:  # try to infer a metadata file
        mdfiles = [f for f in ls if (f.startswith("metadata") and f.endswith(".csv"))]
        if len(mdfiles) != 1:
            raise ValueError("no distinct metadata found")
        metadata_file = op.join(dirname, mdfiles[0])

    if "annotations_file" in kwargs:
        annotations_file = kwargs["annotations_file"]
        if annotations_file != op.abspath(annotations_file):
            annotations_file = op.join(dirname, annotations_file)
    else:  # try to infer annotations file, but don't fail if we can't
        anfiles = [f for f in ls if (f.startswith("annotations") and f.endswith(".csv"))]
        if len(anfiles) == 1:
            annotations_file = op.join(dirname, anfiles[0])
        else:
            annotations_file = None

    if annotations_file is not None and "artificial_units_file" in kwargs:
        artificial_units_file = kwargs["artificial_units_file"]
        if artificial_units_file != op.abspath(artificial_units_file):
            artificial_units_file = op.join(dirname, artificial_units_file)
    else:  # try to infer artificial units file, but don't fail if we can't
        aufiles = [f for f in ls if (f.startswith("artificial_units") and f.endswith(".csv"))]
        if len(aufiles) == 1:
            artificial_units_file = op.join(dirname, aufiles[0])
        else:
            artificial_units_file = None

    if "probe_file" in kwargs:
        probe_file = kwargs["probe_file"]
        if probe_file != op.abspath(probe_file):
            probe_file = op.join(dirname, probe_file)
    else:  # try to infer a probe file
        prfiles = [f for f in ls if (f.startswith("probe") and f.endswith(".npz"))]
        if len(prfiles) != 1:
            raise ValueError("no distinct probe found")
        probe_file = op.join(dirname, prfiles[0])

    if "dtype_file" in kwargs:
        dtype_file = kwargs["dtype_file"]
        if dtype_file != op.abspath(dtype_file):
            dtype_file = op.join(dirname, dtype_file)
    else:  # try to infer a dtype file
        dtfiles = [f for f in ls if (f.startswith("dtype") and f.endswith(".npy"))]
        if len(dtfiles) != 1:
            raise ValueError("no distinct dtype file found")
        dtype_file = op.join(dirname, dtfiles[0])

    # load save data
    metadata = pd.read_csv(metadata_file, index_col=False)
    assert "filename" in metadata.columns and "samples" in metadata.columns and "start_time" in metadata.columns

    # load probe
    probe = load_probe(probe_file)

    # load sample rate and datatype in one go
    sr = np.load(dtype_file)
    sample_rate = sr[0]
    dtype = sr.dtype.type

    dset = DataSet(metadata.filename.values, dtype, sample_rate, probe, metadata.start_time.values)

    if annotations_file is not None:
        annotations = pd.read_csv(annotations_file, index_col=False)
        assert "timestep" in annotations.columns and "cluster" in annotations.columns

        # don't try this at home
        dset.__class__ = AnnotatedDataSet
        dset._annotations = annotations

    if artificial_units_file is not None:
        artificial_units = pd.read_csv(artificial_units_file, index_col=False)
        assert "timestep" in artificial_units.columns and "true_unit" in artificial_units.columns and \
               "center_channel" in artificial_units.columns

        # don't try this at home, either
        dset.__class__ = HybridDataSet
        dset._artificial_units = artificial_units

    return dset


# TODO: this would be great as a single NPZ file...
def save_dataset(dataset, dirname, dataset_name=None):
    """Save a DataSet's metadata to a set of text and binary files.

    Parameters
    ----------
    dataset : DataSet
    dirname : str
    dataset_name : str, optional
    """

    if not isinstance(dataset, DataSet):
        raise TypeError("dataset must be an instance of DataSet")
    elif dataset.isopen and dataset.mode != "r":
        raise ValueError("dataset must be closed or read-only")

    if dataset_name is None:
        dataset_name = int(datetime.datetime.now().timestamp())

    if not op.isdir(dirname):
        os.mkdir(dirname)

    # save metadata : filenames, start times, samples
    mdfile = op.join(dirname, f"metadata-{dataset_name}.csv")
    dataset.export_metadata(mdfile)

    # save annotations : event times, cluster IDs
    if isinstance(dataset, AnnotatedDataSet):
        anfile = op.join(dirname, f"annotations-{dataset_name}.csv")
        dataset.export_annotations(anfile)

    # save artificial units : event times, true units, center channels
    if isinstance(dataset, HybridDataSet):
        aufile = op.join(dirname, f"artificial_units-{dataset_name}.csv")
        dataset.export_artificial_units(aufile)

    # save probe
    prfile = op.join(dirname, f"probe-{dataset_name}.npz")
    save_probe(dataset.probe, prfile)

    # save dtype and sample_rate in one go
    dtfile = op.join(dirname, f"dtype-{dataset_name}.npy")
    np.save(dtfile, np.array([dataset.sample_rate], dtype=dataset.dtype))


class UnopenedDataException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class HybridSourceMismatchError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
