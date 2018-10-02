from context import *

import pandas as pd

from hybridfactory.data import annotation

modbase = op.join(testbase, "annotation")

class TestLoadKilosortRez:
    def setup(self):
        self.ann = annotation.kilosort_from_rez(op.join(modbase, "fromrez"), "eMouse-rez.mat")

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


class TestLoadKilosort2Rez:
    def setup(self):
        self.ann = annotation.kilosort_from_rez(op.join(modbase, "fromrez"), "rez.mat")

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


class TestLoadKilosortPhy: # identical to Kilosort2Rez
    def setup(self):
        self.ann = annotation.kilosort_from_phy(op.join(modbase, "fromphy"))

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


class TestJRCLUSTFromMatfile:
    def setup(self):
        self.ann = annotation.jrclust_from_matfile(op.join(modbase, "fromjrclust"), "testset_jrc.mat")

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


class TestJRCLUSTIncidentals:
    def setup(self):
        self.testdir = op.join(modbase, "fromjrclust")

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


class TestLoadKilosortTemplates:
    def setup(self):
        self.phydir = op.join(modbase, "fromphy")
        self.rezdir = op.join(modbase, "fromrez")

    def test_equivalent(self):
        fromphy = annotation.load_kilosort_templates(self.phydir)
        fromrez = annotation.load_kilosort_templates(self.rezdir)
        assert(np.linalg.norm(fromphy - fromrez) < 1e-5)

    def test_ks1(self):
        fromrez = annotation.load_kilosort_templates(self.rezdir, "eMouse-rez.mat")


class TestLoadGroundTruthMatrix:
    def setup(self):
        self.testdir = op.join(modbase, "fromgroundtruth")

    def test_with_filename(self):
        gt = annotation.load_ground_truth_matrix(self.testdir, "gt.npy")
        assert(gt.size == 0)

    def test_no_filename(self):
        gt = annotation.load_ground_truth_matrix(self.testdir)
        assert(gt.size == 0)


class TestMiscFailures:
    def setup(self):
        self.testdir = op.join(modbase, "miscfailures")

    def test_oldstyle_matfile(self):
        with pytest.raises(ValueError): # contains a v6 MAT file
            annotation.kilosort_from_rez(self.testdir, "oldstyle-rez.mat")

    def test_ambiguous_jrcfile(self):
        with pytest.raises(ValueError): # contains multiple _jrc.mat files
            annotation.jrclust_from_matfile(self.testdir)

    def test_no_source_for_templates(self):
        with pytest.raises(ValueError):
            annotation.load_kilosort_templates(self.testdir)
