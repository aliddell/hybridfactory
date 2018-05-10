import numpy as np

# REQUIRED PARAMETERS

# path to file containing raw source data (currently only SpikeGL-formatted data is supported)
raw_source_file = r"C:\Users\Alan\Documents\Data\eMouse\sim_binary.dat"
# path to file to contain hybrid
raw_target_file = r"C:\Users\Alan\Documents\Data\eMouse\sim_binary_GT.dat"
# type of raw data, as a numpy dtype
data_type = np.int16
# sample rate in Hz
sample_rate = 25000
# directory containing output from spike sorter
data_directory = r"C:\Users\Alan\Documents\Data\eMouse"
# type of output from spike sorter, e.g., "phy", "kilosort" (for rez.mat), "jrclust" (for *.jrc)
output_type = "phy"
# probe layout
probe_type = "eMouse"
# indices (cluster labels) of ground-truth units
ground_truth_units = [8, 16, 18, 26, 32, 34]

# OPTIONAL PARAMETERS

# random seed, for reproducibility
random_seed = 10191
# algorithm to generate hybrid data
generator_type = "steinmetz"
# number of singular values to use in the construction of artificial units
num_singular_values = 6
# number of channels to shift the units by
channel_shift = 4
# standard deviation of time jitter
time_jitter = 500
# number of samples to take before an event timestep
samples_before = 30
# number of samples to take after an event timestep
samples_after = 30
# threshold a channel must exceed to be considered part of an event
event_threshold = -30
# point in the raw file at which the data starts
offset = 0
# whether or not this file contains a sync channel
# (see https://github.com/cortex-lab/neuropixels/wiki/Other_analysis_methods#loading-synchronization-data)
sync_channel = False
# whether or not to copy the source to the target; if not sure, you probably want True
copy = True
# whether or not to overwrite a target file if it already exists
overwrite = True
