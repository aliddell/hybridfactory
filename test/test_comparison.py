from context import *

from hybridfactory.data import dataset as dset
from hybridfactory.generate import generator
from hybridfactory.probes import probe as prb
from hybridfactory.validate import comparison

modbase = op.join(testbase, "comparison")

class TestTemplatePairComparison:
    def setup(self):
        np.random.seed(10191)

        self.testdir = op.join(modbase, "TestTemplatePairComparison")
        # set (and create) directory
        if not op.isdir(self.testdir):
            os.makedirs(self.testdir)
        # link over files if necessary
        self.sourcebin = op.join(self.modbase, "sim_binary.dat")
        if not op.isfile(self.sourcebin):
            os.link(op.join(data_sources, "kilosort1", "eMouse",
                            "sim_binary.dat"), self.sourcebin)
        self.sourcerez = op.join(self.modbase, "rez.mat")
        if not op.isfile(self.sourcebin):
            os.link(op.join(data_sources, "kilosort1", "eMouse",
                            "rez.mat"), self.sourcerez)

        # create a new annotated dataset
        self.dtype = np.int16
        self.sample_rate = 25000
        self.probe = prb.eMouse()
        self.source = dset.new_annotated_dataset(self.filename, self.dtype,
                                                 self.sample_rate, self.probe)

        # create a new hybrid dataset
        hybrid_dir = op.join(self.testdir, "hybrid")
        self.test_unit = 12
        self.hybrid = dset.new_hybrid_dataset(self.source, hybrid_dir,
                                              copy=True)

        # generate a new unit
        svdgen = generator.SVDGenerator(self.hybrid, samples_before=30,
                                        samples_after=30)
        # construct events
        self.events = svdgen.construct_events(self.test_unit, 6)
        # jitter times
        self.event_times = self.hybrid.unit_firing_times(self.test_unit)
        self.jittered_times = svdgen.jitter_events(self.event_times, 100)
        # shift channels
        self.channels = self.hybrid.unit_channels(self.test_unit)
        self.shifted_channels = svdgen.shift_channels(self.channels)

        n_events = self.jittered_times.size
        events = self.events[:, :, np.random.choice(self.events.shape[2],
                                                    n_events, replace=False)]
        svdgen.insert_unit(events, self.jittered_times, self.shifted_channels,
                           true_unit=self.test_unit)
        
        self.hybrid.export_ground_truth_matrix(op.join(self.testdir,
                                                       "firings_true.npy"))

    def teardown(self):
        if op.isfile(self.sourcebin):
            os.unlink(self.sourcebin)
        if op.isfile(self.sourcerez):
            os.unlink(self.sourcerez)