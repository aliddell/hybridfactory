import numpy as np

import os
from os import path as op
import sys

import factory.probes

home = os.getenv("USERPROFILE") if sys.platform == "win32" else os.getenv("HOME")

# REQUIRED PARAMETERS

# directory containing output from spike sorter
data_directory = op.join(home, "Documents", "Data", "eMouse")
# path to file containing raw source data (currently only SpikeGL-formatted data is supported)
raw_source_file = op.join(data_directory, "sim_binary.dat")
# type of raw data, as a numpy dtype
data_type = np.int16
# sample rate in Hz
sample_rate = 25000
# indices (cluster labels) of ground-truth units
ground_truth_units = [6, 17, 20, 36, 45]
# start time for ground-truth firing times
start_time = 0

# PROBE CONFIGURATION

# a probe object; use a prebuilt probe or roll your own
probe = factory.probes.eMouse()

# OPTIONAL PARAMETERS

# session name to identify this run
session_name = "example"
# random seed, for reproducibility
random_seed = 10191
# path to directory in which to output hybrid data file and annotations
output_directory = op.join(data_directory, "hybrid_output")
# type of output from spike sorter, e.g., "phy", "kilosort" (for rez.mat), "jrc" (for *.jrc)
output_type = "phy"
# number of singular values to use in the construction of artificial units
num_singular_values = 6
# number of channels to shift the units by, or None to select at random
channel_shift = None
# synthetic firing rate, in Hz, for hybrid units
synthetic_rate = []
# scale factor for randomly-generated jitter
jitter_factor = 100
# minimum amplitude scale factor
amplitude_scale_min = 0.66
# maximum_amplitude_scale_factor
amplitude_scale_max = 1.5
# number of samples to take before an event timestep
samples_before = 40
# number of samples to take after an event timestep
samples_after = 40
# threshold a channel must exceed to be considered part of an event
event_threshold = -30
# whether or not to copy the source to the target
copy = True
# whether or not to erase the true units from the hybrid data (EXPERIMENTAL)
erase = True
