from context import *

from hybridfactory.data import dataset
from hybridfactory.probes import probe
from hybridfactory.generate import generator

class TestGenerateHybridJRCLUST:
    def setup(self):
        self.testdir = op.join(testbase, "generator", "fromjrclust")
        self.filename = op.join(self.testdir, "anm420712_20180802_ch0-119bank1_ch120-382bank0_g0_t2.imec.ap.bin")
        self.dtype = np.int16
        self.sample_rate = 30000
        self.probe = probe.neuropixels3a()
        self.source = dataset.new_annotated_dataset(self.filename, self.dtype,
                                                 self.sample_rate, self.probe)
        hybrid_dir = op.join(self.testdir, "hybrid")

        self.hybrid = dataset.new_hybrid_dataset(self.source, hybrid_dir, copy=False)

    def test_svd_generator(self):
        dset = self.hybrid
        event_threshold = -290
        samples_before = 17
        samples_after = 23
        svdgen = generator.SVDGenerator(dset, event_threshold, samples_before, samples_after)
        events = svdgen.construct_events(1291, 3)

        assert(events.shape == (4, 41, 572))
