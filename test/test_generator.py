from context import *

import scipy.spatial.distance

from hybridfactory.data import dataset as dset
from hybridfactory.probes import probe as prb
from hybridfactory.generate import generator

class TestSVDGenerator:
    def setup(self):
        self.testdir = op.join(testbase, "generator", "fromjrclust")
        self.filename = op.join(self.testdir, "anm420712_20180802_ch0-119bank1_ch120-382bank0_g0_t2.imec.ap.bin")
        self.dtype = np.int16
        self.sample_rate = 30000
        self.probe = prb.neuropixels3a()
        self.source = dset.new_annotated_dataset(self.filename, self.dtype,
                                                 self.sample_rate, self.probe)
        self.test_unit = 309

        hybrid_dir = op.join(self.testdir, "hybrid")

        self.hybrid = dset.new_hybrid_dataset(self.source, hybrid_dir, copy=True)
        self.svdgen = generator.SVDGenerator(self.hybrid, samples_before=40, samples_after=40)
        # construct events
        self.events = self.svdgen.construct_events(self.test_unit, 3)
        # jitter times
        self.event_times = self.hybrid.unit_firing_times(self.test_unit)
        self.jittered_times = self.svdgen.jitter_events(self.event_times, 100)
        # shift channels
        self.channels = self.hybrid.unit_channels(self.test_unit)
        self.shifted_channels = self.svdgen.shift_channels(self.channels)

    def test_construct_events(self):
        assert(self.events.shape == (4, 81, 2476))

    def test_scale_events(self):
        scaled_events = self.svdgen.scale_events(self.events)
        assert((scaled_events[:, 0, :] == 0).all())
        assert((scaled_events[:, 0, :] == scaled_events[:, -1, :]).all())

    def test_jitter_times(self):
        timediffs = np.abs(self.jittered_times[:, np.newaxis] - self.event_times[np.newaxis, :]).ravel()
        # consistently expect a vanishingly small number of events in the interspike interval
        assert(np.count_nonzero(timediffs < self.hybrid.sample_rate/1000)/timediffs.size < 1e-5)
        assert(self.jittered_times.size <= self.events.shape[2])

    def test_synthetic_times(self):
        times = self.svdgen.synthetic_firing_times(60, 0)
        assert(times.size == 36000)
        assert((np.diff(np.sort(times)) == 500).all())

    def test_shift_channels(self):
        channel_pdist = scipy.spatial.distance.pdist(self.hybrid.probe.channel_coordinates(self.channels))
        shifted_pdist = scipy.spatial.distance.pdist(self.hybrid.probe.channel_coordinates(self.shifted_channels))
        assert(np.isclose(channel_pdist, shifted_pdist).all())

        # fixed channel shift
        shifted_channels = self.svdgen.shift_channels(self.channels, 100)
        channel_pdist = scipy.spatial.distance.pdist(self.hybrid.probe.channel_coordinates(self.channels))
        shifted_pdist = scipy.spatial.distance.pdist(self.hybrid.probe.channel_coordinates(shifted_channels))
        assert(np.isclose(channel_pdist, shifted_pdist).all())

    def test_insert_units(self):
        jittered_times = self.jittered_times
        shifted_channels = self.shifted_channels
        hybrid = self.hybrid
        test_unit = self.test_unit

        n_events = jittered_times.size
        events = self.events[:, :, np.random.choice(self.events.shape[2], n_events, replace=False)]
        self.svdgen.insert_unit(events, jittered_times, shifted_channels, true_unit=test_unit)
        assert((hybrid.artificial_units.timestep == np.sort(jittered_times)).all())
        assert((hybrid.artificial_units.true_unit == test_unit).all())

    def test_reset(self):
        self.hybrid.reset(self.source)
        assert(md5sum(self.filename) == md5sum(self.hybrid.filenames[0]))
