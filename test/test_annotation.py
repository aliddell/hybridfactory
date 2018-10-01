from context import *

import pandas as pd

from hybridfactory.data import annotation

class TestLoadKilosortRez:
    def setup(self):
        self.ann = annotation.kilosort_from_rez(op.join(testbase, "annotation", "fromrez"), "eMouse-rez.mat")

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
        self.ann = annotation.kilosort_from_rez(op.join(testbase, "annotation", "fromrez"), "ks2-rez.mat")

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
        self.ann = annotation.kilosort_from_phy(op.join(testbase, "annotation", "fromphy"))

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
        self.ann = annotation.jrclust_from_matfile(op.join(testbase, "annotation", "fromjrclust"))

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
        self.testdir = op.join(testbase, "annotation", "fromjrclust")

    def test_load_features(self):
        features = annotation.load_jrc_features(self.testdir)
        assert(np.abs(features[4, 1, 10190] - 6.6172601e+02) < 1e-5)

    def test_load_filtered(self):
        filtered = annotation.load_jrc_filtered(self.testdir)
        assert(filtered[16, 5, 10190] == -19)

    def test_load_raw(self):
        raw = annotation.load_jrc_raw(self.testdir)
        assert(raw[38, 3, 10190] == 424)


class TestLoadKilosortTemplates:
    def setup(self):
        self.phydir = op.join(testbase, "annotation", "fromphy")
        self.rezdir = op.join(testbase, "annotation", "fromrez")

    def test_equivalent(self):
        fromphy = annotation.load_kilosort_templates(self.phydir)
        fromrez = annotation.load_kilosort_templates(self.rezdir, "ks2-rez.mat")
        assert(np.linalg.norm(fromphy - fromrez) < 1e-5)
