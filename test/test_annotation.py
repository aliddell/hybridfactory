from context import *

import pandas as pd

from hybridfactory.data import annotation

modbase = op.join(testbase, "annotation")

class TestLoadKilosortRez:
    def setup(self):
        # set (and create) directory
        self.testdir = op.join(modbase, "TestLoadKilosortRez")
        if not op.isdir(self.testdir):
            os.makedirs(self.testdir)
        # link over files if necessary
        sourcefile = op.join(data_sources, "kilosort1", "eMouse", "rez.mat")
        self.testfile = op.join(self.testdir, "rez.mat")
        if not op.isfile(self.testfile):
            os.link(sourcefile, self.testfile)
        
        self.ann = annotation.kilosort_from_rez(self.testdir, "rez.mat")

    def test_load_success(self):
        assert(isinstance(self.ann, pd.DataFrame))

    def test_num_spikes(self):
        assert(self.ann.shape[0] == 168879)

    def test_clusters(self):
        clusters = self.ann.cluster
        assert(clusters.unique().size == 31)
        # clusters are 1-based, since they aren't used as indices
        assert(clusters.min() == 0) # BUT there is a garbage cluster
        assert(clusters.max() == 64)
        assert(clusters[10190] == 6)

    def test_templates(self):
        templates = self.ann.template
        assert(templates.unique().size == 52)
        # templates are 0-based, since they **are** used as indices
        assert(templates.min() == 0)
        assert(templates.max() == 63)
        assert(templates[10190] == 5)

    def test_timesteps(self):
        timesteps = self.ann.timestep
        assert(timesteps[0] == timesteps.min() == 129)
        assert(timesteps[timesteps.last_valid_index()] == timesteps.max() == 24999766)
        assert(timesteps.unique().size == 168310)
        
    def teardown(self):
        if op.isfile(self.testfile):
            os.unlink(self.testfile)


class TestLoadKilosort2Rez:
    def setup(self):
        # set (and create) directory
        self.testdir = op.join(modbase, "TestLoadKilosort2Rez")
        if not op.isdir(self.testdir):
            os.makedirs(self.testdir)
        # link over files if necessary
        sourcefile = op.join(data_sources, "kilosort2", "Hopkins", "rez.mat")
        self.testfile = op.join(self.testdir, "rez.mat")
        if not op.isfile(self.testfile):
            os.link(sourcefile, self.testfile)
        
        self.ann = annotation.kilosort_from_rez(self.testdir, "rez.mat")

    def test_load_success(self):
        assert(isinstance(self.ann, pd.DataFrame))

    def test_num_spikes(self):
        assert(self.ann.shape[0] == 8938169)

    def test_clusters(self):
        clusters = self.ann.cluster
        assert(clusters.unique().size == 347)
        # clusters are 1-based, since they aren't used as indices
        # this dataset hasn't been manually curated
        assert(clusters.min() == 1)
        assert(clusters.max() == 347)
        assert(clusters[10190] == 81)

    def test_templates(self):
        templates = self.ann.template
        assert(templates.unique().size == 347)
        # templates are 0-based, since they **are** used as indices
        assert(templates.min() == 0)
        assert(templates.max() == 346)
        assert(templates[10190] == 80)

    def test_timesteps(self):
        timesteps = self.ann.timestep
        assert(timesteps[0] == timesteps.min() == 135)
        assert(timesteps[timesteps.last_valid_index()] == timesteps.max() == 113208737)
        assert(timesteps.unique().size == 8548226)
        
    def teardown(self):
        if op.isfile(self.testfile):
            os.unlink(self.testfile)


class TestLoadKilosortPhy: # identical to Kilosort2Rez
    def setup(self):
        # set (and create) directory
        self.testdir = op.join(modbase, "TestLoadKilosortPhy")
        if not op.isdir(self.testdir):
            os.makedirs(self.testdir)
        # link over files if necessary
        self.testfiles = []
        for fn in ("spike_times.npy", "spike_clusters.npy",
                   "spike_templates.npy"):
            sourcefile = op.join(data_sources, "kilosort2", "Hopkins", fn)
            self.testfiles.append(op.join(self.testdir, fn))

            if not op.isfile(self.testfiles[-1]):
                os.link(sourcefile, self.testfiles[-1])
        
        self.ann = annotation.kilosort_from_phy(self.testdir)

    def test_load_success(self):
        assert(isinstance(self.ann, pd.DataFrame))

    def test_num_spikes(self):
        assert(self.ann.shape[0] == 8938169)

    def test_clusters(self):
        clusters = self.ann.cluster
        assert(clusters.unique().size == 347)
        # clusters are 1-based, since they aren't used as indices
        # this dataset hasn't been manually curated
        assert(clusters.min() == 1)
        assert(clusters.max() == 347)
        assert(clusters[10190] == 81)

    def test_templates(self):
        templates = self.ann.template
        assert(templates.unique().size == 347)
        # templates are 0-based, since they **are** used as indices
        assert(templates.min() == 0)
        assert(templates.max() == 346)
        assert(templates[10190] == 80)

    def test_timesteps(self):
        timesteps = self.ann.timestep
        assert(timesteps[0] == timesteps.min() == 135)
        assert(timesteps[timesteps.last_valid_index()] == timesteps.max() == 113208737)
        assert(timesteps.unique().size == 8548226)
        
    def teardown(self):
        for fn in self.testfiles:
            if op.isfile(fn):
                os.unlink(fn)


class TestJRCLUSTFromMatfile:
    def setup(self):
        # set (and create) directory
        self.testdir = op.join(modbase, "TestJRCLUSTFromMatfile")
        if not op.isdir(self.testdir):
            os.makedirs(self.testdir)
        # link over files if necessary
        sourcefile = op.join(data_sources, "jrclust", "testset", "testset_jrc.mat")
        self.testfile = op.join(self.testdir, "testset_jrc.mat")
        if not op.isfile(self.testfile):
            os.link(sourcefile, self.testfile)
        
        self.ann = annotation.jrclust_from_matfile(self.testdir, "testset_jrc.mat")

    def test_load_success(self):
        assert(isinstance(self.ann, pd.DataFrame))

    def test_num_spikes(self):
        assert(self.ann.shape[0] == 4302921)

    def test_clusters(self):
        clusters = self.ann.cluster
        assert(clusters.unique().size == 200)
        # clusters are 1-based, since they aren't used as indices
        assert(clusters.min() == 0) # BUT there is a garbage cluster
        assert(clusters.max() == 199)
        assert(clusters[10190] == 186)

    def test_channels(self):
        channels = self.ann.channel_index
        assert(channels.unique().size == 64)
        # channels are 0-based, since they **are** used as indices
        assert(channels.min() == 0)
        assert(channels.max() == 63)
        assert(channels[10190] == 54)

    def test_timesteps(self):
        timesteps = self.ann.timestep
        assert(timesteps[0] == timesteps.min() == 125)
        assert(timesteps[timesteps.last_valid_index()] == timesteps.max() == 104478015)
        assert(timesteps.unique().size == 4214174)
        
    def teardown(self):
        if op.isfile(self.testfile):
            os.unlink(self.testfile)


class TestJRCLUSTIncidentals:
    def setup(self):
        # set (and create) directory
        self.testdir = op.join(modbase, "TestJRCLUSTIncidentals")
        if not op.isdir(self.testdir):
            os.makedirs(self.testdir)
        # link over files if necessary
        self.testfiles = []
        for fn in ("testset_jrc.mat", "testset_spkfet.jrc",
                   "testset_spkwav.jrc", "testset_spkraw.jrc"):
            sourcefile = op.join(data_sources, "jrclust", "testset", fn)
            self.testfiles.append(op.join(self.testdir, fn))

            if not op.isfile(self.testfiles[-1]):
                os.link(sourcefile, self.testfiles[-1])

    def test_load_features(self):
        features = annotation.load_jrc_features(self.testdir, "testset_spkfet.jrc")
        assert(np.abs(features[4, 1, 10190] - 6.6172601e+02) < 1e-5)

        features2 = annotation.load_jrc_features(self.testdir)
        assert(np.linalg.norm(features - features2) == 0)

    def test_load_filtered(self):
        filtered = annotation.load_jrc_filtered(self.testdir, "testset_spkwav.jrc")
        assert(filtered[16, 5, 10190] == -19)

        filtered2 = annotation.load_jrc_filtered(self.testdir)
        assert(np.linalg.norm(filtered - filtered2) == 0)

    def test_load_raw(self):
        raw = annotation.load_jrc_raw(self.testdir, "testset_spkraw.jrc")
        assert(raw[38, 3, 10190] == 424)

        raw2 = annotation.load_jrc_raw(self.testdir)
        assert(np.linalg.norm(raw - raw2) == 0)
        
    def teardown(self):
        for fn in self.testfiles:
            if op.isfile(fn):
                os.unlink(fn)


class TestLoadKilosortTemplates:
    def setup(self):
        # set (and create) directories
        self.testdir = op.join(modbase, "TestLoadKilosortTemplates")
        self.phydir = op.join(self.testdir, "fromphy")
        self.rezdir = op.join(self.testdir, "fromrez")

        if not op.isdir(self.phydir):
            os.makedirs(self.phydir)
        if not op.isdir(self.rezdir):
            os.makedirs(self.rezdir)

        # link over files if necessary
        self.testfiles = [op.join(self.rezdir, "rez.mat"), # Hopkins rez
                          op.join(self.phydir, "templates.npy"),
                          op.join(self.testdir, "rez.mat")] # eMouse rez

        if not op.isfile(self.testfiles[0]):
            os.link(op.join(data_sources, "kilosort2", "Hopkins", "rez.mat"),
                    self.testfiles[0])
        if not op.isfile(self.testfiles[1]):
            os.link(op.join(data_sources, "kilosort2", "Hopkins",
                            "templates.npy"), self.testfiles[1])
        if not op.isfile(self.testfiles[2]):
            os.link(op.join(data_sources, "kilosort1", "eMouse",
                            "rez.mat"), self.testfiles[2])

    def test_equivalent(self):
        fromphy = annotation.load_kilosort_templates(self.phydir)
        fromrez = annotation.load_kilosort_templates(self.rezdir)
        assert(np.linalg.norm(fromphy - fromrez) < 1e-5)

    def test_ks1(self):
        fromrez = annotation.load_kilosort_templates(self.testdir, "rez.mat")
        
    def teardown(self):
        for fn in self.testfiles:
            if op.isfile(fn):
                os.unlink(fn)


class TestLoadGroundTruthMatrix:
    def setup(self):
        # set (and create) directory
        self.testdir = op.join(modbase, "TestLoadGroundTruthMatrix")
        if not op.isdir(self.testdir):
            os.makedirs(self.testdir)
        # link over files if necessary
        self.testfiles = [op.join(self.testdir, "firings_true.npy"),
                          op.join(self.testdir, "gt.npy")]

        if not op.isfile(self.testfiles[0]):
            os.link(op.join(data_sources, "groundtruth", "firings_true.npy"),
                    self.testfiles[0])
        if not op.isfile(self.testfiles[1]):
            os.link(op.join(data_sources, "groundtruth", "firings_true.npy"),
                            self.testfiles[1])

    def test_with_filename(self):
        gt = annotation.load_ground_truth_matrix(self.testdir, "gt.npy")
        assert(gt.size == 0)

    def test_no_filename(self):
        gt = annotation.load_ground_truth_matrix(self.testdir)
        assert(gt.size == 0)
        
    def teardown(self):
        for fn in self.testfiles:
            if op.isfile(fn):
                os.unlink(fn)


class TestMiscFailures:
    def setup(self):
        # set (and create) directory
        self.testdir = op.join(modbase, "TestMiscFailures")
        if not op.isdir(self.testdir):
            os.makedirs(self.testdir)
        # link over files if necessary
        self.testfiles = [op.join(self.testdir, "rez.mat"), # v6 MAT file
                          op.join(self.testdir, "testset1_jrc.mat"),
                          op.join(self.testdir, "testset2_jrc.mat")]

        if not op.isfile(self.testfiles[0]):
            os.link(op.join(data_sources, "misc", "oldstyle-rez.mat"),
                    self.testfiles[0])
        if not op.isfile(self.testfiles[1]):
            os.link(op.join(data_sources, "jrclust", "testset",
                            "testset_jrc.mat"), self.testfiles[1])
        if not op.isfile(self.testfiles[2]):
            os.link(op.join(data_sources, "jrclust", "testset",
                            "testset_jrc.mat"), self.testfiles[2])

    def test_oldstyle_matfile(self):
        with pytest.raises(ValueError): # contains a v6 MAT file
            annotation.kilosort_from_rez(self.testdir, "rez.mat")

    def test_ambiguous_jrcfile(self):
        with pytest.raises(ValueError): # contains multiple _jrc.mat files
            annotation.jrclust_from_matfile(self.testdir)

    def test_no_source_for_templates(self):
        with pytest.raises(ValueError):
            annotation.load_kilosort_templates(self.testdir)
            
    def teardown(self):
        for fn in self.testfiles:
            if op.isfile(fn):
                os.unlink(fn)
