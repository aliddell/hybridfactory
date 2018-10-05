from context import *

import pandas as pd

from hybridfactory.probes import probe as prb
from hybridfactory.data import dataset as dset

modbase = op.join(testbase, "dataset")

class TestAnnotatedDatasetJRCLUST:
    def setup(self):
        self.testdir = op.abspath(op.join(modbase, "fromjrclust"))
        self.filename = op.join(self.testdir, "anm420712_20180802_ch0-119bank1_ch120-382bank0_g0_t2.imec.ap.bin")
        self.dtype = np.int16
        self.sample_rate = 30000
        self.probe = prb.neuropixels3a()
        self.dataset = dset.new_annotated_dataset(self.filename, self.dtype,
                                                  self.sample_rate, self.probe)

    def test_load_success(self):
        dataset = self.dataset

        assert(dataset.dtype == self.dtype)

        assert(len(dataset.filenames) == 1)
        assert(op.abspath(dataset.filenames[0]) == op.abspath(self.filename))

        assert(not dataset.isopen)

        metadata = dataset.metadata
        assert(metadata.shape[0] == 1)
        assert(op.abspath(metadata.filename[0]) == op.abspath(self.filename))
        assert(metadata.start_time[0] == 0)
        assert(metadata.samples[0] == dataset.last_sample() == 18000000)

        assert(dataset.mode is None)

        assert(dataset.probe == self.probe == prb.neuropixels3a())

        assert(dataset.sample_rate == 30000)

        assert((dataset.start_times == pd.Series([0])).all())

    def test_annotations(self):
        ann = self.dataset.annotations
        assert(isinstance(ann, pd.DataFrame))
        assert(ann.shape[0] == 1433951)

        clusters = ann.cluster
        assert(clusters.unique().size == 1336)
        # clusters are 1-based, since they aren't used as indices
        assert(clusters.min() == 0) # BUT there is a garbage cluster
        assert(clusters.max() == 1335)
        assert(clusters[10190] == 731)

        channels = ann.channel_index
        assert(channels.unique().size == 384)
        # channels are 0-based, since they **are** used as indices
        assert(channels.min() == 0)
        assert(channels.max() == 383)
        assert(channels[10190] == 204)

        timesteps = ann.timestep
        assert(timesteps[0] == timesteps.min() == 3)
        assert(timesteps[timesteps.last_valid_index()] == timesteps.max() == 17999957)
        assert(timesteps.unique().size == 1333051)

    def test_open_close_raw(self):
        dataset = self.dataset
        dataset.open_raw()

        assert(dataset.isopen)
        assert(dataset.mode == "r")

        dataset.close_raw()

        assert(not dataset.isopen)
        assert(dataset.mode is None)

    def test_read_write_roi(self):
        dataset = self.dataset
        dataset.open_raw("r+")

        channels = np.arange(385)
        samples = np.arange(1000,2000)

        # read a known ROI
        roi = dataset.read_roi(channels, samples)
        assert(roi[0, 0] == -309)
        assert(np.abs(np.linalg.norm(roi, ord='fro') - 1.091067674207242e+05) < 1e-10)

        # generate a new ROI and write it to this region
        perturbation = np.random.randint(-6, 6, roi.shape)
        new_roi = roi + perturbation
        dataset.write_roi(channels, samples, new_roi)
        assert((dataset.read_roi(channels, samples) == new_roi).all())

        # restore the old ROI to its rightful place
        dataset.write_roi(channels, samples, roi)
        assert((dataset.read_roi(channels, samples) == roi).all())

        dataset.close_raw()

    def test_unit_annotations(self):
        dataset = self.dataset

        assert(dataset.unit_event_count(1) == 339)
        assert(dataset.unit_event_count(10191) == 0)
        assert(dataset.unit_event_count("foo") == 0)

        firing_times = dataset.unit_firing_times(1)
        assert(firing_times[0] == firing_times.min() == 174452)
        assert(firing_times[-1] == firing_times.max() == 17845107)
        assert(len(dataset.unit_firing_times("foo")) == 0)

        assert((np.unique(dataset.unit_channels(1)) == np.array([0, 1, 2, 3, 4, 5])).all())

    def test_hybrid_dataset(self):
        source = self.dataset

        hybrid_dir = op.join(self.testdir, "hybrid")
        hds = dset.new_hybrid_dataset(source, hybrid_dir, copy=False)

        assert(op.basename(hds.filenames[0]) == op.basename(self.filename).replace(".bin", ".GT.bin"))
        assert(len(hds.artificial_units) == 0)

        hds.export_artificial_units(op.join(hybrid_dir, "au.csv"))
        assert(pd.read_csv(op.join(hybrid_dir, "au.csv")).shape[0] == 0)

        hds.export_ground_truth_matrix(op.join(hybrid_dir, "gt.npy"))
        gtu = np.load(op.join(hybrid_dir, "gt.npy"))
        assert(gtu.size == 0)

        hds.reset(source)

    def test_save_load(self):
        dataset = self.dataset

        dset.save_dataset(dataset, self.testdir, dataset_name="test-save")

        mdf = "metadata-test-save.csv"
        anf = "annotations-test-save.csv"
        prf = "probe-test-save.npz"
        dtf = "dtype-test-save.npy"

        dload = dset.load_dataset(self.testdir, metadata_file=mdf, annotations_file=anf,
                                  probe_file=prf, dtype_file=dtf)

        assert(isinstance(dload, dset.AnnotatedDataSet))
        assert(dload.dtype == dataset.dtype)
        assert(dload.probe == dataset.probe)
        assert((dload.metadata == dataset.metadata).all().all())
        assert((dload.annotations == dataset.annotations).all().all())

class TestAnnotatedDatasetKilosort:
    def setup(self):
        self.testdir = op.abspath(op.join(modbase, "fromkilosort"))
        self.filename = op.join(self.testdir, "sim_binary.dat")
        self.dtype = np.int16
        self.sample_rate = 25000
        self.probe = prb.eMouse()
        self.dataset = dset.new_annotated_dataset(self.filename, self.dtype,
                                                  self.sample_rate, self.probe)

    def test_load_success(self):
        dataset = self.dataset

        assert(dataset.dtype == self.dtype)

        assert(len(dataset.filenames) == 1)
        assert(op.abspath(dataset.filenames[0]) == op.abspath(self.filename))

        assert(not dataset.isopen)

        metadata = dataset.metadata
        assert(metadata.shape[0] == 1)
        assert(op.abspath(metadata.filename[0]) == op.abspath(self.filename))
        assert(metadata.start_time[0] == 0)
        assert(metadata.samples[0] == dataset.last_sample() == 25000000)

        assert(dataset.mode is None)

        assert(dataset.probe == self.probe == prb.eMouse())

        assert(dataset.sample_rate == 25000)

        assert((dataset.start_times == pd.Series([0])).all())

    def test_annotations(self):
        ann = self.dataset.annotations
        assert(isinstance(ann, pd.DataFrame))
        assert(ann.shape[0] == 168879)

        clusters = ann.cluster
        assert(clusters.unique().size == 31)
        # clusters are 1-based, since they aren't used as indices
        assert(clusters.min() == 0) # BUT there is a garbage cluster
        assert(clusters.max() == 64)
        assert(clusters[10190] == 6)

        templates = ann.template
        assert(templates.unique().size == 52)
        # templates are 0-based, since they **are** used as indices
        assert(templates.min() == 0)
        assert(templates.max() == 63)
        assert(templates[10190] == 5)

        timesteps = ann.timestep
        assert(timesteps[0] == timesteps.min() == 129)
        assert(timesteps[timesteps.last_valid_index()] == timesteps.max() == 24999766)
        assert(timesteps.unique().size == 168310)

    def test_read_write_roi(self):
        dataset = self.dataset
        dataset.open_raw("r+")

        channels = dataset.probe.connected_channels()
        samples = np.arange(1000,2000)

        # read a known ROI
        roi = dataset.read_roi(channels, samples)
        assert(roi[0, 0] == -534)
        assert(np.abs(np.linalg.norm(roi, ord='fro') - 3.633788725283846e+04) < 1e-10)

        # generate a new ROI and write it to this region
        perturbation = np.random.randint(-6, 6, roi.shape)
        new_roi = roi + perturbation
        dataset.write_roi(channels, samples, new_roi)
        assert((dataset.read_roi(channels, samples) == new_roi).all())

        # restore the old ROI to its rightful place
        dataset.write_roi(channels, samples, roi)
        assert((dataset.read_roi(channels, samples) == roi).all())

        dataset.close_raw()

    def test_unit_annotations(self):
        dataset = self.dataset

        assert(dataset.unit_event_count(1) == 9016)
        assert(dataset.unit_event_count(10191) == 0)
        assert(dataset.unit_event_count("foo") == 0)

        firing_times = dataset.unit_firing_times(1)
        assert(firing_times[0] == firing_times.min() == 2460)
        assert(firing_times[-1] == firing_times.max() == 24999342)
        assert(len(dataset.unit_firing_times("foo")) == 0)

        assert((np.unique(dataset.unit_channels(1)) == np.array([0, 18, 20, 22, 24, 26, 28])).all())

    def test_hybrid_dataset(self):
        source = self.dataset

        hybrid_dir = op.join(self.testdir, "hybrid")
        hds = dset.new_hybrid_dataset(source, hybrid_dir, copy=False)

        assert(op.basename(hds.filenames[0]) == op.basename(self.filename).replace(".dat", ".GT.dat"))
        assert(len(hds.artificial_units) == 0)

        hds.export_artificial_units(op.join(hybrid_dir, "au.csv"))
        assert(pd.read_csv(op.join(hybrid_dir, "au.csv")).shape[0] == 0)

        hds.export_ground_truth_matrix(op.join(hybrid_dir, "gt.npy"))
        gtu = np.load(op.join(hybrid_dir, "gt.npy"))
        assert(gtu.size == 0)

        hds.reset(source)

    def test_save_load(self):
        dataset = self.dataset

        dset.save_dataset(dataset, self.testdir, dataset_name="test-save")

        mdf = "metadata-test-save.csv"
        anf = "annotations-test-save.csv"
        prf = "probe-test-save.npz"
        dtf = "dtype-test-save.npy"

        dload = dset.load_dataset(self.testdir, metadata_file=mdf, annotations_file=anf,
                                  probe_file=prf, dtype_file=dtf)

        assert(isinstance(dload, dset.AnnotatedDataSet))
        assert(dload.dtype == dataset.dtype)
        assert(dload.probe == dataset.probe)
        assert((dload.metadata == dataset.metadata).all().all())
        assert((dload.annotations == dataset.annotations).all().all())


class TestMiscFailures:
    def setup(self):
        self.testdir = op.join(modbase, "miscfailures")

    def test_dataset_constructor(self):
        filename = op.join(self.testdir, "sim_binary.dat")
        dtype = np.int16
        sample_rate = 25000
        probe = prb.eMouse()
        start_times = 0

        with pytest.raises(TypeError): # dtype fails
            dset.DataSet(filename, "np.int16", sample_rate, probe, start_times)

        with pytest.raises(TypeError): # probe fails
            dset.DataSet(filename, dtype, sample_rate, "probe", start_times)

        with pytest.raises(TypeError): # sample rate fails: integer
            dset.DataSet(filename, dtype, float(sample_rate), probe, start_times)

        with pytest.raises(ValueError): # sample rate fails: positive
            dset.DataSet(filename, dtype, -sample_rate, probe, start_times)

        with pytest.raises(IOError): # filenames
            dset.DataSet(None, dtype, sample_rate, probe, start_times)

        with pytest.raises(ValueError): # filenames/start times mismatch
            dset.DataSet(filename, dtype, sample_rate, probe, [start_times, 0])

        with pytest.raises(TypeError): # start times fails: integer
            dset.DataSet(filename, dtype, sample_rate, probe, 0.0)

        with pytest.raises(ValueError): # start times fails: positive
            dset.DataSet(filename, dtype, sample_rate, probe, -1)
