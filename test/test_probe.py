from context import *

from hybridfactory import probes

class TestEmouse:
    def setup(self):
        self.probe = probes.eMouse()

    def test_getters(self):
        probe = self.probe
        assert(probe.num_channels == 34)
        assert((probe.channel_map[probe.inverse_channel_map] == np.arange(34)).all())
        assert((probe.connected == np.array([False, False, True, True, True, True,
                                             True, True, True, True, True, True, True,
                                             True, True, True, True, True, True, True,
                                             True, True, True, True, True, True, True,
                                             True, True, True, True, True, True, True])).all())

        assert(probe.name == "eMouse")

    def test_setters(self):
        probe = self.probe

        # channel_map
        with pytest.raises(TypeError): # wrong datatype
            probe.channel_map = "foo"
        with pytest.raises(ValueError): # wrong **array** datatype (float)
            probe.channel_map = np.zeros((34,))
        with pytest.raises(ValueError): # wrong shape
            probe.channel_map = np.zeros((10, 1, 91), dtype=np.int64)
        with pytest.raises(ValueError): # wrong channel count
            probe.channel_map = probe.connected_channels()

        old_chanmap = probe.channel_map
        probe.channel_map = np.arange(1, 35)
        assert((probe.channel_map == np.arange(1,35)).all())
        with pytest.raises(ValueError): # immutable
            probe.channel_map[0] = 9

        # reset
        probe.channel_map = old_chanmap

        # connected
        with pytest.raises(TypeError): # wrong datatype
            probe.connected = "foo"
        with pytest.raises(ValueError): # wrong **array** datatype (float)
            probe.connected = np.zeros((34,))
        with pytest.raises(ValueError): # wrong shape
            probe.connected = np.zeros((10, 1, 91), dtype=np.bool)
        with pytest.raises(ValueError): # wrong channel count
            probe.connected = np.zeros((35,), dtype=np.bool)

        old_connected = probe.connected
        probe.connected = np.ones((34,), dtype=np.bool)
        assert(probe.connected.all())
        probe.connected[0] = False # mutable
        assert(not probe.connected.all())
        # reset
        probe.connected = old_connected

        # channel_positions
        with pytest.raises(TypeError): # wrong datatype
            probe.channel_positions = "foo"
        with pytest.raises(ValueError): # wrong dimension count
            probe.channel_positions = np.zeros((10, 1, 91))
        with pytest.raises(ValueError): # wrong channel count
            probe.channel_positions = probe.connected_positions()
        with pytest.raises(ValueError): # wrong shape (< 2 columns)
            probe.channel_positions = np.zeros((34, 1))

        old_chanpos = probe.channel_positions
        probe.channel_positions = np.zeros((34, 2))
        assert(np.linalg.norm(probe.channel_positions) == 0.0)
        with pytest.raises(ValueError): # immutable
            probe.channel_positions[0, 1] = 9

        # reset
        probe.channel_positions = old_chanpos

        # name
        with pytest.raises(TypeError):
            probe.name = 0

        old_name = probe.name
        probe.name = "Child-like Empress"
        assert(probe.name == "Child-like Empress")


    def test_comparison(self):
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
        cprobe = probes.custom_probe(channel_map, connected, channel_positions, "eMouse")

        assert(self.probe == probes.eMouse() == cprobe)


    def test_methods(self):
        probe = self.probe
        assert(np.isnan(probe.channel_coordinates([32, 33])).all())
        assert((probe.connected_channels() == np.array([7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27,
                                                        29, 31, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
                                                        26, 28, 30, 0, 1, 2, 3, 4, 5])).all())
        assert((probe.connected_positions() == 20*np.hstack((np.array([1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                                                                      0, 1, 0, 1, 0, 1, 0])[:, np.newaxis],
                                                            np.array([7, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 17, 17, 18,
                                                                      18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24])[:, np.newaxis]))).all())
        assert((probe.coordinate_channels(np.array([[20, 140]])) == np.array([7])).all())
        assert(len(probe.display()) == 2)
        assert(len(probe.possible_shifts([1, 2])) == 13)
        assert(len(probe.possible_shifts([1, 2, 5, 6, 9, 10])) == 0)


    def test_save_load(self):
        probe = self.probe
        probes.save_probe(probe, op.join(testbase, "probes", "test_save.npz"))
        newprobe = probes.load_probe(op.join(testbase, "probes", "test_save.npz"))
        assert(probe == newprobe)

        # clean up
        os.unlink(op.join(testbase, "probes", "test_save.npz"))

        with pytest.raises(TypeError):
            probes.save_probe("foo", op.join(testbase, "probes", "test_failsave.npz"))

        with pytest.raises(ValueError):
            probes.load_probe(op.join(testbase, "probes", "not_a_probe.npz"))

class TestNeuropixels3A:
    def setup(self):
        self.probe = probes.neuropixels3a()

    def test_equals(self):
        assert(self.probe != probes.neuropixels3a(False))
