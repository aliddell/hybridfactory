# Copyright (C) 2018 Vidrio Technologies. All rights reserved.

import abc
from collections import OrderedDict
import numbers

import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.spatial
import scipy.stats

from hybridfactory.data.dataset import HybridDataSet


class HybridEventGenerator(object):
    def __init__(self, dataset):
        if not isinstance(dataset, HybridDataSet):
            raise TypeError("dataset must be an instance of HybridDataSet")

        self._dataset = dataset

    @abc.abstractmethod
    def construct_events(self, unit):
        pass


class SVDGenerator(HybridEventGenerator):
    def __init__(self, dataset, samples_before, samples_after):
        super().__init__(dataset)

        if not isinstance(samples_before, numbers.Integral):
            raise TypeError("samples_before must be an integer")
        elif samples_before <= 0:
            raise ValueError("samples_before must be positive")

        self._samples_before = samples_before

        if not isinstance(samples_after, numbers.Integral):
            raise TypeError("samples_after must be an integer")
        elif samples_after <= 0:
            raise ValueError("samples_after must be positive")

        self._samples_after = samples_after

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, val):
        if isinstance(val, HybridDataSet):
            self._dataset = val

    @property
    def probe(self):
        return self._dataset.probe

    @property
    def samples_after(self):
        return self._samples_after

    @property
    def samples_before(self):
        return self._samples_before

    def construct_events(self, unit, num_singular_values, num_events=None):
        """

        Parameters
        ----------
        unit : int
            Cluster ID of unit to construct events from.
        num_singular_values : int
            Number of singular values to use in reconstruction of events.
        num_events : int, optional
            Number of events desired.

        Returns
        -------
        recon_events : numpy.ndarray
            Tensor, num_channels x num_samples x num_events, artificially-constructed events.
        """

        if not isinstance(num_singular_values, numbers.Integral):
            raise TypeError("num_singular_values must be an integer")
        elif num_singular_values <= 0:
            raise ValueError("num_singular_values must be positive")

        if num_events is not None and not isinstance(num_events, numbers.Integral):
            raise TypeError("num_events must be an integer")
        elif num_events is not None and num_events <= 0:
            raise ValueError("num_events must be positive")

        if not self._dataset.isopen:
            close_after = True
            self._dataset.open_raw("r")
        else:
            close_after = False

        events = self._dataset.unit_windows(unit, self.samples_before, self.samples_after)
        if np.prod(events.shape) == 0:
            raise NoEventException(f"no events found for unit {unit}")

        if min(np.prod(events.shape[:2]), events.shape[2]) < num_singular_values:
            raise RankDeficientException(f"not enough events or samples for {num_singular_values} singular values")

        # now create sub-array for just appropriate channels
        channels = self._dataset.unit_channels(unit=unit,
                                               samples_before=self._samples_before,
                                               samples_after=self._samples_after)

        if self._samples_before + self._samples_after + 1 < channels.size:
            raise RankDeficientException("fewer samples given than channels")

        events = events[channels, :, :]

        num_channels, num_samples, num_original_events = events.shape

        if num_events is None:
            num_events = num_original_events

        events_shift = events - events[:, 0, :].reshape(num_channels, 1, num_original_events, order="F")

        scale = np.arange(num_samples).reshape(1, num_samples, 1, order="F") / (num_samples - 1)
        events_detrended = events_shift - events_shift[:, -1, :].reshape(num_channels, 1, num_original_events,
                                                                         order="F") * scale
        events_diff = np.diff(events_detrended, axis=1)

        # compute the SVD on the derivative of the waveforms
        flat_spikes = events_diff.reshape(num_channels * (num_samples - 1), num_original_events, order="F")
        u, s, vt = np.linalg.svd(flat_spikes, full_matrices=True)

        # take just the most significant singular values
        u = np.matrix(u[:, :num_singular_values])
        s = np.matrix(np.diag(s[:num_singular_values]))
        vt = np.matrix(vt[:num_singular_values, :])

        # reconstruct artificial event derivatives from SVD and integrate
        flat_recon_events = np.array(u * s * vt)
        recon_events = np.hstack(
            (np.zeros((num_channels, 1, num_original_events)),
             np.cumsum(flat_recon_events.reshape(num_channels, num_samples - 1, num_original_events, order="F"),
                       axis=1))
        )

        if close_after:
            self._dataset.close_raw()

        return recon_events[:, :, np.random.choice(num_original_events, size=num_events, replace=True)]

    # @staticmethod
    # def erase_events(events, centers, kind="cubic"):
    #     """Remove events from windows via spline interpolation.
    #
    #     Parameters
    #     ----------
    #     events : numpy.ndarray
    #         Tensor, num_channels x num_samples x num_events.
    #     centers : numpy.ndarray
    #         Array, num_events. Event centers in windows.
    #     kind : str, optional
    #
    #     Returns
    #     -------
    #     erased_events : numpy.ndarray
    #         Events, after erasure.
    #     """
    #
    #     def _find_left_right(win, center):
    #         assert win.ndim == 1
    #
    #         m = win.size // 2
    #         rad = win.size // 8
    #
    #         if center is None:  # assume window is "centered" at event
    #             nbd = win[m - rad:m + rad]
    #             center = np.abs(nbd).argmax() + m - rad
    #             is_local_min = nbd[rad] == nbd.min()
    #         else:
    #             nbd = win[center - rad:center + rad]
    #             is_local_min = nbd[rad] == nbd.min()
    #
    #         # find points where first and second derivative change signs
    #         wdiff = np.diff(win)  # first derivative
    #         scale_factor = np.abs(wdiff)
    #         scale_factor[scale_factor == 0] = 1
    #         abswdiff = wdiff / scale_factor
    #
    #         wdiff2 = np.diff(wdiff)  # second derivative
    #         scale_factor2 = np.abs(wdiff2)
    #         scale_factor2[scale_factor2 == 0] = 1
    #         abswdiff2 = wdiff2 / scale_factor2
    #
    #         turning_points = np.union1d(
    #                 np.where(np.array([abswdiff[i] != abswdiff[i + 1] for i in range(abswdiff.size - 1)]))[0] + 1,
    #                 np.where(np.array([abswdiff2[i] != abswdiff2[i + 1] for i in range(abswdiff2.size - 1)]))[0] + 2)
    #         tp_center = np.abs(center - turning_points).argmin()
    #
    #         if tp_center < 2 or tp_center > turning_points.size - 2:
    #             if is_local_min:  # all differences are negative until center, then positive
    #                 # find the last difference before the center that is positive
    #                 wleft = center - np.where(wdiff[:center - 1][::-1] > 0)[0][0]
    #                 # find the first difference after the center that is negative
    #                 wright = center + np.where(wdiff[center + 1:] < 0)[0][0]
    #             else:  # all differences are positive until center, then negative
    #                 # find the last difference before the center that is negative
    #                 wleft = center - np.where(wdiff[:center - 1][::-1] < 0)[0][0]
    #                 # find the first difference after the center that is positive
    #                 wright = center + np.where(wdiff[center + 1:] > 0)[0][0]
    #         else:
    #             wleft = turning_points[tp_center - 2] + (center - turning_points[tp_center - 2]) // 2
    #             wright = turning_points[tp_center + 2] + (turning_points[tp_center + 2] - center) // 2
    #
    #         return wleft, wright
    #
    #     if centers is None:
    #         centers = events.min(axis=0).argmin(axis=0)
    #     elif not isinstance(centers, np.ndarray):
    #         raise TypeError("centers must be a NumPy array")
    #     elif not np.issubdtype(centers.dtype, np.integer):
    #         raise TypeError("centers must be an integral array")
    #     elif centers.size != events.shape[2]:
    #         raise ValueError("centers length must match num_events")
    #
    #     erased_events = events.copy().astype(np.float64)
    #
    #     for k in range(events.shape[2]):
    #         event = erased_events[:, :, k]
    #
    #         for j, channel_window in enumerate(event):
    #             wl, wr = _find_left_right(channel_window, center=centers[k])
    #
    #             exes = np.hstack((np.arange(wl), np.arange(wr, channel_window.size)))
    #             whys = channel_window[exes]
    #             g = scipy.interpolate.interp1d(exes, whys, kind)
    #
    #             event[j, :] = g(np.arange(channel_window.size))
    #             event[j, wl:wr] += 3 * np.random.randn(wr - wl)
    #
    #     return erased_events.astype(events.dtype)

    # def erase_unit(self, unit, kind="cubic"):
    #     """Remove all events for a given unit.
    #
    #     Parameters
    #     ----------
    #     unit : int
    #         Cluster ID of unit to erase events from.
    #     kind : str, optional
    #     """
    #
    #     events = self._dataset.unit_windows(unit, self._samples_before, self._samples_after)
    #     event_times = self._dataset.unit_firing_times(unit)
    #
    #     channels = self._dataset.unit_channels(unit=unit,
    #                                            samples_before=self._samples_before,
    #                                            samples_after=self._samples_after)
    #
    #     erased_events = self.erase_events(events[channels, :], centers=np.tile(self._samples_before, events.shape[2]),
    #                                       kind=kind)
    #
    #     # write data to disk
    #     for k, t in enumerate(event_times):
    #         samples = np.arange(t - self._samples_before, t + self._samples_after + 1)
    #         self._dataset.write_roi(channels, samples, erased_events[:, :, k])
    #
    #     # erase record from annotation
    #     ann = self._dataset.annotations
    #     self._dataset._annotations = ann.drop(ann[ann.cluster == unit].index)

    def insert_unit(self, events, event_times, channels, true_unit, new_unit=None):
        """Insert a new unit into the data.

        Parameters
        ----------
        events : numpy.ndarray
            Tensor, num_channels x num_samples x num_events.
        event_times : numpy.ndarray
            Array of firing times for new unit.
        channels : numpy.ndarray
            Channels in which to insert new unit.
        true_unit : int
        new_unit : int, optional
        """

        if not isinstance(events, np.ndarray):
            raise TypeError("events must be NumPy array")
        elif events.ndim != 3:
            raise ValueError("shape mismatch")

        if not isinstance(event_times, np.ndarray):
            raise TypeError("events must be NumPy array")
        elif event_times.size != events.shape[2]:
            raise ValueError("time/event mismatch")
        elif (event_times < 0).any():
            raise ValueError("negative times")

        if not isinstance(channels, np.ndarray):
            raise TypeError("channels must be NumPy array")
        elif not np.isin(channels, self.probe.channel_map).all():
            raise ValueError("channels not in channel map")
        elif channels.size != events.shape[0]:
            raise ValueError("number of channels given does not match event shape")

        if new_unit is not None and not isinstance(new_unit, numbers.Integral):
            raise TypeError("unit_id must be an integer")
        elif new_unit is not None and new_unit in self._dataset.annotations.cluster:
            raise ValueError("unit_id must not refer to an already existing cluster")

        if events.dtype != self._dataset.dtype:
            events = events.astype(self._dataset.dtype)

        if new_unit is None:
            new_unit = self._dataset.annotations.cluster.max() + 1

        if not self._dataset.isopen:
            close_after = True
            self.dataset.open_raw("r+")
        else:
            close_after = False

        # write data to disk
        for k, t in enumerate(event_times):
            samples = np.arange(t - self._samples_before, t + self._samples_after + 1)
            old_data = self._dataset.read_roi(channels, samples)
            self._dataset.write_roi(channels, samples, events[:, :, k] + old_data)

        # add new record to annotation
        ann = self._dataset.annotations
        if "template" in ann.columns:
            stack = pd.DataFrame(data={"timestep": event_times,
                                       "cluster": new_unit,
                                       "template": ann.template.max() + 1})
        else:
            stack = pd.DataFrame(data={"timestep": event_times,
                                       "cluster": new_unit,
                                       "channel_index": -1})

        ann = ann.append(stack)
        ann.sort_values("timestep", inplace=True)
        ann.index = np.arange(ann.shape[0])

        self._dataset._annotations = ann

        # add new record to artificial_units
        center_channel = scipy.stats.mode(events[:, self._samples_before, :].argmin(axis=0)).mode[0]
        au = self._dataset.artificial_units

        stack = pd.DataFrame(data=OrderedDict([("timestep", event_times),
                                               ("true_unit", true_unit),
                                               ("new_unit", new_unit),
                                               ("center_channel", center_channel)]))

        au = au.append(stack)
        au.sort_values("timestep", inplace=True)
        au.index = np.arange(au.shape[0])

        self._dataset._artificial_units = au

        if close_after:
            self._dataset.close_raw()

    def jitter_events(self, event_times, jitter_factor, isi=2):
        """

        Parameters
        ----------
        event_times : numpy.ndarray
            Firing times for these events, to be jittered.
        jitter_factor : int
            Standard deviation of number of samples to jitter by.
        isi : int, optional
            Interspike interval, in milliseconds. Window around true event times to avoid when jittering.

        Returns
        -------
        jittered_times : numpy.ndarray
            Jittered firing times for artificial events.
        """

        if not isinstance(event_times, np.ndarray):
            raise TypeError("event_times must be a NumPy array")
        elif not np.issubdtype(event_times.dtype, np.integer):
            raise ValueError("event_times must be integers")
        elif (event_times < 0).any():
            raise ValueError("event_times must be nonnegative")

        isi_half = self.dataset.sample_rate * isi // 2000  # number of samples in `isi/2` ms
        j_scale = (jitter_factor // 2)

        # normally-distributed jitter factor, with an absmin of `isi_half`
        jitter1 = isi_half + np.abs(np.random.normal(loc=0, scale=j_scale, size=event_times.size // 2))
        jitter2 = -(isi_half + np.abs(isi_half + np.random.normal(loc=0, scale=j_scale,
                                                                  size=event_times.size - jitter1.size)))

        # leaves a window of `isi` ms around `event_times` so units don't fire right on top of each other
        jitter = np.random.permutation(np.hstack((jitter1, jitter2))).astype(event_times.dtype)
        jittered_times = event_times + jitter

        # ensure ROIs don't fall outside of data files
        window_start = self.dataset.metadata.start_time.values
        window_stop = window_start + self.dataset.metadata.samples.values

        # find file into which t - samples_before goes...
        before_index = np.searchsorted(window_start, jittered_times - self._samples_before, side="right")
        # ...and ensure that it's the same file that t + samples_after goes
        after_index = np.searchsorted(window_stop, jittered_times + self._samples_after, side="left") + 1

        mask = before_index == after_index
        jittered_times = jittered_times[mask]

        return jittered_times

    @staticmethod
    def scale_events(events, scale_min=0.98, scale_max=1.02):
        """Randomly scale events.

        Parameters
        ----------
        events : numpy.ndarray
            Tensor, num_channels x num_samples x num_events.
        scale_min: float
            Minimum scale factor.
        scale_max: float
            Maximum scale factor.

        Returns
        -------
        scaled_events : numpy.ndarray
            Tensor, num_channels x num_samples x num_events, scaled.
        """

        centers = np.abs(events).max(axis=0).argmax(axis=0)
        scale_factors = np.random.uniform(scale_min, scale_max,
                                          size=events.shape[2])
        scale_rows = []
        for k in range(events.shape[2]):
            sr = np.hstack((np.linspace(0, scale_factors[k], centers[k]),
                            np.linspace(scale_factors[k], 0, events.shape[1] - centers[k] + 1)[1:]))[np.newaxis, :]

            scale_rows.append(sr)

        return np.stack(scale_rows, axis=2) * events

    def shift_channels(self, channels, shift_factor=None):
        """Shift a subset of channels.

        Parameters
        ----------
        channels : numpy.ndarray
            Input channels to be shifted.
        shift_factor : int, optional
            Constant shift factor.

        Returns
        -------
        shifted_channels : numpy.ndarray or None
            Channels shifted by some factor.
        """

        if isinstance(channels, numbers.Integral):
            channels = np.array([channels])
        elif hasattr(channels, "__iter__"):
            channels = np.array(channels)
        else:
            raise TypeError("channels must be an integer or an iterable of integers")

        if not np.issubdtype(channels.dtype, np.integer):
            raise ValueError("channels must be an integer or an iterable of integers")

        # inverse_channel_map[probe.channel_map] == [1, 2, ..., probe.num_channels - 1]
        inverse_channel_map = self.probe.inverse_channel_map

        if shift_factor is not None:
            # make sure our shifted channels fall in the range [0, probe.channel_map)
            if inverse_channel_map[channels].max() < self.probe.channel_map.size - shift_factor:
                shifted_channels = self.probe.channel_map[inverse_channel_map[channels] + shift_factor]
            elif inverse_channel_map[channels].min() >= shift_factor:
                shifted_channels = self.probe.channel_map[inverse_channel_map[channels] - shift_factor]
            else:  # give up
                shift_factor = None

            if shift_factor is not None:
                # channel shift places events outside of probe range
                if not (shifted_channels.min() > -1 and shifted_channels.max() < self.probe.channel_map.size):
                    shift_factor = None

                # channel shift places events on unconnected channels
                if np.intersect1d(shifted_channels, self.probe.channel_map[~self.probe.connected]).size != 0:
                    shift_factor = None

                channel_distance = scipy.spatial.distance.pdist(
                        self.probe.channel_positions[inverse_channel_map[channels], :])
                shifted_distance = scipy.spatial.distance.pdist(
                        self.probe.channel_positions[inverse_channel_map[shifted_channels], :])

                # channel shift alters spatial relationship between channels
                if not np.isclose(channel_distance, shifted_distance).all():
                    shift_factor = None

        # shift factor was originally None or manual shift failed
        if shift_factor is None:
            shift_candidates = self.probe.possible_shifts(channels)
            if not shift_candidates:
                return np.nan + np.zeros_like(channels)

            # bias in favor of channels further out from anchor
            weights = 1 + np.arange(len(shift_candidates), dtype=np.float64)
            weights /= weights.sum()
            anchor = np.random.choice(list(shift_candidates.keys()), p=weights)

            shifted_channels = shift_candidates[anchor]

        return shifted_channels

    def synthetic_firing_times(self, firing_rate, jitter_factor=100):
        """

        Parameters
        ----------
        firing_rate : int
            Synthetic firing rate, in Hz.
        jitter_factor : int, optional
            Standard deviation of number of samples to jitter by.

        Returns
        -------
            Firing times for synthetic units.
        """

        if not isinstance(firing_rate, numbers.Integral):
            raise TypeError("firing_rate must be an integer")
        elif firing_rate <= 0 or firing_rate > self.dataset.sample_rate:
            raise ValueError("firing_rate must be between 0 and sample rate")

        if not isinstance(jitter_factor, numbers.Integral):
            raise TypeError("jitter_factor must be an integer")
        elif jitter_factor < 0 or (jitter_factor > self.dataset.metadata.samples.values).any():
            raise ValueError("jitter_factor must be between 0 and sample rate")

        start = self._samples_before
        stop = self.dataset.last_sample() - self._samples_after - 1
        step = self.dataset.sample_rate // firing_rate

        synthetic_times = np.arange(start=start, stop=stop, step=step)

        return self.jitter_events(synthetic_times, jitter_factor, isi=0)


class NoEventException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RankDeficientException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
