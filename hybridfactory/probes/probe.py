# Copyright (C) 2018 Vidrio Technologies. All rights reserved.

from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance


class Probe(object):
    def __init__(self, channel_map, connected, positions, name="Custom"):
        """

        Parameters
        ----------
        channel_map : numpy.ndarray
        connected : numpy.ndarray
        positions : numpy.ndarray
        name : str, optional
        """
        if not isinstance(channel_map, np.ndarray):
            raise TypeError("channel_map must be a NumPy array")
        elif not np.issubdtype(channel_map.dtype, np.integer):
            raise ValueError("channel_map must be an integer type")
        elif np.ravel(channel_map).ndim != 1:
            raise ValueError("channel_map must be 1-dimensional")

        # channel_map is fixed
        self._channel_map = np.ravel(channel_map.copy())
        self._channel_map.setflags(write=False)

        if not isinstance(connected, np.ndarray):
            raise TypeError("connected must be a NumPy array")
        elif not np.issubdtype(connected.dtype, np.bool_):
            raise ValueError("connected must be a boolean type")
        elif np.ravel(connected).ndim != 1:
            raise ValueError("connected must be 1-dimensional")
        elif connected.size != channel_map.size:
            raise ValueError("connected and channel_map must be the same size")

        # connected is NOT fixed
        self._connected = np.ravel(connected.copy())

        if not isinstance(positions, np.ndarray):
            raise TypeError("channel_positions must be a NumPy array")
        elif positions.ndim != 2:
            raise ValueError("channel_positions must be 2-dimensional")
        elif positions.shape[0] != channel_map.size:
            raise ValueError("first dimension of channel_positions must match size of channel_map")
        elif positions.shape[1] < 2:
            raise ValueError("second dimension of channel_positions must have length at least 2")

        # channel_positions is fixed
        self._channel_positions = positions.copy()
        self._channel_positions.setflags(write=False)

        if not isinstance(name, str):
            raise TypeError("name must be a string")

        self._name = name

    def __repr__(self):
        return f"{self._name} Probe"

    def __hash__(self):
        cm = tuple(self._channel_map.tolist())
        cn = tuple(self._connected.tolist())
        ps = tuple(self._channel_positions.ravel().tolist())

        return hash((cm, cn, ps))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__hash__() == other.__hash__()

    @property
    def num_channels(self):
        return self._channel_map.size

    @property
    def channel_map(self):
        return self._channel_map

    @channel_map.setter
    def channel_map(self, val):
        if not isinstance(val, np.ndarray):
            raise TypeError("channel_map must be a NumPy array")
        elif not np.issubdtype(val.dtype, np.integer):
            raise ValueError("channel_map must be an integer type")
        elif np.squeeze(val).ndim != 1:
            raise ValueError("channel_map must be 1-dimensional")
        elif val.size != self.num_channels:
            raise ValueError("assignment cannot alter number of channels")

        self._channel_map = np.ravel(val.copy())
        self._channel_map.setflags(write=False)

    @property
    def connected(self):
        return self._connected

    @connected.setter
    def connected(self, val):
        if not isinstance(val, np.ndarray):
            raise TypeError("connected must be a NumPy array")
        elif not np.issubdtype(val.dtype, np.bool_):
            raise ValueError("connected must be a boolean type")
        elif np.squeeze(val).ndim != 1:
            raise ValueError("connected must be 1-dimensional")
        elif val.size != self.num_channels:
            raise ValueError("assignment cannot alter number of channels")

        self._connected = np.ravel(val.copy())

    @property
    def channel_positions(self):
        return self._channel_positions

    @channel_positions.setter
    def channel_positions(self, val):
        if not isinstance(val, np.ndarray):
            raise TypeError("channel_positions must be a NumPy array")
        elif val.ndim != 2:
            raise ValueError("channel_positions must be 2-dimensional")
        elif val.shape[0] != self.num_channels:
            raise ValueError("assignment cannot alter number of channels")
        elif val.shape[1] < 2:
            raise ValueError("second dimension of channel_positions must have length at least 2")

        self._channel_positions = val.copy()
        self._channel_positions.setflags(write=False)

    @property
    def inverse_channel_map(self):
        # inverse_channel_map[probe.channel_map] == [1, 2, ..., probe.num_channels - 1]
        icm = np.zeros_like(self.channel_map)
        icm[self.channel_map] = np.arange(self.channel_map.size)

        return icm

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val):
        if not isinstance(val, str):
            raise TypeError("name must be a string")

        self._name = val

    def channel_coordinates(self, channels):
        """Get coordinates of `channels`.

        Parameters
        ----------
        channels : numpy.ndarray

        Returns
        -------
        coords : numpy.ndarray
        """

        if not hasattr(channels, "__iter__"):
            channels = np.array([channels])
        elif not isinstance(channels, np.ndarray):
            channels = np.array(channels)

        assert channels.min() >= 0 and channels.max() <= self._channel_map.max()

        return self.channel_positions[self.inverse_channel_map[channels], :]

    def connected_channels(self):
        return self._channel_map[self._connected]

    def connected_positions(self):
        return self._channel_positions[self._connected, :]

    def coordinate_channels(self, coords):
        """Get channel IDs at `coords`.

        Parameters
        ----------
        coords : numpy.ndarray

        Returns
        -------
        channels : numpy.ndarray
        """
        assert coords[:, 0].min() >= 0 and coords[:, 0].max() <= self.connected_positions()[:, 0].max()
        assert coords[:, 1].min() >= 0 and coords[:, 1].max() <= self.connected_positions()[:, 1].max()

        channels = np.zeros(coords.shape[0], dtype=self._channel_map.dtype)

        for i, xy in enumerate(coords):
            channels[i] = self._channel_map[np.all(self._channel_positions == xy, axis=1)][0]

        return channels

    def display(self, label=True, *args, **kwargs):
        """Return a plot of the probe channel_positions.

        Returns
        -------
        fig : matplotlib.Figure
        ax : matplotlib.AxesSubplot
        """
        fig = plt.figure()
        ax = fig.gca()

        x, y = self._channel_positions[self._connected, :].T
        ax.scatter(x, y, *args, **kwargs)

        if label:
            for k, xy in enumerate(zip(x + 0.1, y + 0.1)):
                ax.annotate(f"{self.connected_channels()[k]}", xy=xy, textcoords='data')

        ax.grid()

        return fig, ax

    def possible_shifts(self, subset):
        def _furthest_from_center(channels):
            probe_center = self.connected_positions().mean(axis=0)[np.newaxis, :]

            pos = self.channel_coordinates(channels)
            ai = scipy.spatial.distance.cdist(probe_center, pos).ravel().argmax()
            ank = channels[ai]

            return ank

        # probe_positions = self.connected_positions()
        # probe_channels = self.connected_channels()

        subset_coords = self.channel_coordinates(subset)
        subset_pdist = scipy.spatial.distance.pdist(subset_coords)

        anchor = _furthest_from_center(subset)
        anchor_coords = self.channel_coordinates(anchor)
        dist_relations = subset_coords - anchor_coords

        matches = {}
        for candidate in self.connected_channels():
            if candidate == anchor:
                continue
            candidate_dist_relations = self.connected_positions() - self.channel_coordinates(candidate)
            candidate_channels = -np.ones_like(subset)

            for i, dr in enumerate(dist_relations):
                dr_matches = np.all(candidate_dist_relations == dr, axis=1)
                if not dr_matches.any():  # no match in this relation
                    break
                else:
                    candidate_channels[i] = self.connected_channels()[dr_matches][0]
            # did we survive?
            if (candidate_channels == -1).any():
                continue
            else:

                assert np.isclose(scipy.spatial.distance.pdist(self.channel_coordinates(candidate_channels)),
                                  subset_pdist).all()
                matches[candidate] = candidate_channels

        if not matches:
            return matches

        # now sort these in order
        match_keys = np.array(list(matches.keys()))
        match_dists = scipy.spatial.distance.cdist(anchor_coords, self.channel_coordinates(match_keys)).ravel()

        match_order = match_dists.argsort()

        return OrderedDict([(k, matches[k]) for k in match_keys[match_order]])


def custom_probe(channel_map, connected, channel_positions, name="Custom"):
    """

    Parameters
    ----------
    channel_map : numpy.ndarray
        Map of channel ID to row in raw data.
    connected : numpy.ndarray
        Boolean array, i is True if channel_map[i] is an active channel.
    channel_positions : numpy.ndarray
        2d array, row[i] corresponds to xy position of channel_map[i].
    name : str, optional
        A name for the probe.

    Returns
    -------
    probe : Probe
    """

    return Probe(channel_map, connected, channel_positions, name)


def neuropixels3a(sync_channel=True):
    """Create and return a Neuropixels Phase 3A probe.

    (See https://github.com/cortex-lab/neuropixels/wiki/About_Neuropixels for details.)

    Returns
    -------
    probe : Probe
    """
    num_channels = 384 + int(sync_channel)  # 384 channels + sync channel

    # the full channel map for the Neuropixels Phase 3A array
    channel_map = np.arange(num_channels)

    # reference channels
    refchans = np.array([36, 75, 112, 151, 188, 227, 264, 303, 340, 379, 384])
    connected = ~np.isin(channel_map, refchans)

    # physical location of each channel on the probe
    xcoords = np.hstack((np.tile([43, 11, 59, 27], num_channels // 4)))  # 43 11 59 27 43 11 59 27 ...
    ycoords = 20 * np.hstack((np.repeat(1 + np.arange(num_channels // 2), 2)))  # 20 20 40 40 ... 3820 3820 3840 3840
    channel_positions = np.hstack((xcoords[:, np.newaxis], ycoords[:, np.newaxis]))
    if sync_channel:
        channel_positions = np.vstack((channel_positions, np.nan*np.ones((1,2))))

    return Probe(channel_map, connected, channel_positions, name="Neuropixels Phase 3A")


def eMouse():
    """Create and return an eMouse probe.

    (See https://github.com/cortex-lab/KiloSort/tree/master/eMouse for details.)

    Returns
    -------
    probe : Probe
    """
    channel_map = np.array([32, 33, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27,
                            29, 31, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
                            26, 28, 30, 0, 1, 2, 3, 4, 5], dtype=np.int64)

    # reference channels
    refchans = np.array([32, 33])
    connected = ~np.isin(channel_map, refchans)

    # physical location of each channel on the probe
    xcoords = 20 * np.array([np.nan, np.nan, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                             0, 1, 0, 1, 0, 1, 0])
    ycoords = 20 * np.array([np.nan, np.nan, 7, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 17, 17, 18,
                             18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24])
    channel_positions = np.hstack((xcoords[:, np.newaxis], ycoords[:, np.newaxis]))

    return Probe(channel_map, connected, channel_positions, name="eMouse")


def save_probe(probe, filename):
    """

    Parameters
    ----------
    probe : Probe
    filename : str

    Returns
    -------

    """

    if not isinstance(probe, Probe):
        raise TypeError("probe must be an instance of Probe")

    np.savez(filename, channel_map=probe.channel_map, connected=probe.connected,
             channel_positions=probe.channel_positions, name=probe.name)


def load_probe(filename):
    """

    Parameters
    ----------
    filename : str

    Returns
    -------

    """

    probez = np.load(filename)

    try:
        channel_map = probez["channel_map"]
        connected = probez["connected"]
        positions = probez["channel_positions"]
        name = str(probez["name"])
    except KeyError:
        raise ValueError(f"{filename} does not contain a probe")

    return custom_probe(channel_map, connected, positions, name)
