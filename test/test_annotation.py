from context import *

import pandas as pd

from hybridfactory.data import annotation


# @testset "Import a sorting from KiloSort rez.mat" begin
#     rezfile = joinpath(ENV["TESTBASE"], "datasets", "fromrez", "eMouse-rez.mat")
#     sorting = sortingfromrezfile(rezfile, programversion="0fbe8eb", runtimesecs=0,
#                                  modelname="eMouse", recordedby="Shagrat", sortedby="Gorbag")
#     chanmap = [8; 10; 12; 14; 16; 18; 20; 22; 24; 26; 28; 30; 32; 7; 9; 11; 13; 15; 17; 19; 21; 23; 25; 27; 29; 31; 1; 2; 3; 4; 5; 6]
#     xc = [20; 0; 0; 20; 0; 20; 0; 20; 0; 20; 0; 20; 0; 20; 0; 20; 20; 0; 20; 0; 20; 0; 20; 0; 20; 0; 20; 0; 20; 0; 20; 0]
#     yc = [140; 160; 180; 180; 200; 200; 220; 220; 240; 240; 260; 260; 280; 280; 300; 300; 320; 340; 340; 360; 360; 380; 380; 400; 400; 420; 420; 440; 440; 460; 460; 480]
#     chanpos = [xc yc]
#
#     @test channelmap(dataset(autoannotation(sorting)).probe) == chanmap
#     @test channelpositions(dataset(autoannotation(sorting)).probe) == chanpos
#     @test nchannels(dataset(autoannotation(sorting)).probe) == 34
#
#     @test length(channelmap(dataset(ksannotation).probe)) == 32
#     @test size(channelpositions(dataset(ksannotation).probe)) == (32, 2)
#     @test minimum(amplitudes(ksannotation)) ≈ 10.027755737304688
#     @test maximum(amplitudes(ksannotation)) ≈ 1.145809936523438e+02
#     @test minimum(similartemplates(ksannotation)[CartesianIndex.(1:64, 1:64)]) == 0
#     @test maximum(similartemplates(ksannotation)[CartesianIndex.(1:64, 1:64)]) ≈ 1.0000008 # !!
#     @test spiketemplates(ksannotation)[10191] == 6
# end
#
# @testset "Import a sorting from KiloSort2 rez.mat" begin
#     rezfile = joinpath(ENV["TESTBASE"], "datasets", "fromrez", "ks2-rez.mat")
#     sorting = sortingfromrezfile(rezfile, programversion="77bd485", runtimesecs=0,
#                                  modelname="Neuropixels Phase 3A", recordedby="Shagrat",
#                                  sortedby="Gorbag")
#     chanmap = setdiff(1:385, [37 76 113 152 189 228 265 304 341 380 385])
#     chanpos = [repeat([0], 374) chanmap]
#
#     @test channelmap(dataset(autoannotation(sorting)).probe) == chanmap
#     @test channelpositions(dataset(autoannotation(sorting)).probe) == chanpos
#     @test modelname(dataset(autoannotation(sorting)).probe) == "Neuropixels Phase 3A"
#     @test nchannels(dataset(autoannotation(sorting)).probe) == 385
#
#     ksannotation = autoannotation(sorting)
#     @test nclusters(ksannotation) == 347
#     @test length(channelmap(dataset(ksannotation).probe)) == 374
#     @test size(channelpositions(dataset(ksannotation).probe)) == (374, 2)
#     @test minimum(amplitudes(ksannotation)) ≈ 12.018919944763184
#     @test maximum(amplitudes(ksannotation)) ≈ 106.52466583251953
#     @test minimum(similartemplates(ksannotation)[CartesianIndex.(1:347, 1:347)]) ≈ 0.9829312
#     @test maximum(similartemplates(ksannotation)[CartesianIndex.(1:347, 1:347)]) ≈ 0.98877007
#     @test spiketemplates(ksannotation)[10191] == 81
# end

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
