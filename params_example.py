import numpy as np

import os
from os import path as op
import sys

import hybridfactory.probes

# your home directory
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
channel_map = np.array([32, 33, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27,
                        29, 31, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
                        26, 28, 30, 0, 1, 2, 3, 4, 5], dtype=np.int64)

# reference channels
refchans = np.array([32, 33])
connected = ~np.isin(channel_map, refchans)

# physical location of each channel on the probe
xcoords = 20 * np.array([np.nan, np.nan, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                         1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
ycoords = 20 * np.array([np.nan, np.nan, 7, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
                         13, 14, 14, 15, 15, 16, 17, 17, 18, 18, 19, 19, 20, 20,
                         21, 21, 22, 22, 23, 23, 24])
channel_positions = np.hstack((xcoords[:, np.newaxis], ycoords[:, np.newaxis]))
probe = hybridfactory.probes.custom_probe(channel_map, connected,
                                          channel_positions, name="eMouse")
# equivalently:
# probe = hybridfactory.probes.eMouse()

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
# whether or not to copy the source to the target
copy = True
