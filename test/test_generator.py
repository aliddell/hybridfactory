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

    def construct_events(self):
        self.events = svdgen.construct_events(test_unit, 3)
        assert(self.events.shape == (4, 81, 2476))

    def test_scale_events(self):
        events = self.events

        # scale events
        scaled_events = svdgen.scale_events(events)
        assert((scaled_events[:, 0, :] == 0).all())
        assert((scaled_events[:, 0, :] == scaled_events[:, -1, :]).all())

        # jitter times
        event_times = hybrid.unit_firing_times(test_unit)
        jittered_times = svdgen.jitter_events(event_times, 100)
        timediffs = np.abs(jittered_times[:, np.newaxis] - event_times[np.newaxis, :]).ravel()
        # a vanishingly small number of events in the interspike interval
        assert(np.count_nonzero(timediffs < hybrid.sample_rate/1000)/timediffs.size < 1e-5)
        assert(jittered_times.size <= events.shape[2])

        # generate synthetic firing times
        times = svdgen.synthetic_firing_times(60, 0)
        assert(times.size == 36000)
        assert((np.diff(np.sort(times)) == 500).all())

        # shift channels
        channels = hybrid.unit_channels(test_unit)
        shifted_channels = svdgen.shift_channels(channels, 100)
        channel_pdist = scipy.spatial.distance.pdist(hybrid.probe.channel_coordinates(channels))
        shifted_pdist = scipy.spatial.distance.pdist(hybrid.probe.channel_coordinates(shifted_channels))
        assert(np.isclose(channel_pdist, shifted_pdist).all())

        # get random channel shift
        shifted_channels = svdgen.shift_channels(channels)
        channel_pdist = scipy.spatial.distance.pdist(hybrid.probe.channel_coordinates(channels))
        shifted_pdist = scipy.spatial.distance.pdist(hybrid.probe.channel_coordinates(shifted_channels))
        assert(np.isclose(channel_pdist, shifted_pdist).all())

        # insert units
        n_events = jittered_times.size
        events = events[:, :, np.random.choice(events.shape[2], n_events, replace=False)]
        svdgen.insert_unit(events, jittered_times, shifted_channels, true_unit=test_unit)
        assert((hybrid.artificial_units.timestep == np.sort(jittered_times)).all())
        assert((hybrid.artificial_units.true_unit == test_unit).all())

        # reset
        hybrid.reset(self.source)
        assert(md5sum(self.filename), md5sum(hybrid.filename[0]))
