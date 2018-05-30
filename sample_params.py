import numpy as np

# REQUIRED PARAMETERS

# path to file containing raw source data (currently only SpikeGL-formatted data is supported)
raw_source_file = r"C:\Users\Alan\Documents\Data\eMouse\sim_binary.dat"
# path to file to contain hybrid data
raw_target_file = r"C:\Users\Alan\Documents\Data\eMouse\sim_binary_GT.dat"
# start from an empty (+ noisy) file if True
from_empty = True
# type of raw data, as a numpy dtype
data_type = np.int16
# sample rate in Hz
sample_rate = 25000
# directory containing output from spike sorter
data_directory = r"C:\Users\Alan\Documents\Data\eMouse"
# type of output from spike sorter, e.g., "phy", "kilosort" (for rez.mat), "jrc" (for *.jrc)
output_type = "phy"
# probe layout
probe_type = "eMouse"
# indices (cluster labels) of ground-truth units
ground_truth_units = [6, 17, 20, 36, 45, 57]
# start time for ground-truth firing times
start_time = 0

# OPTIONAL PARAMETERS

# random seed, for reproducibility
random_seed = 10191
# algorithm to generate hybrid data
generator_type = "svd_generator"
# number of singular values to use in the construction of artificial units
num_singular_values = 6
# number of channels to shift the units by, or None to select at random
channel_shift = None
# synthetic firing rate, in Hz, for hybrid units
synthetic_rate = [100, 150, 200, 250, 300, 350]
# scale factor for randomly-generated jitter
jitter_factor = 100
# minimum amplitude scale factor
amplitude_scale_min = 0.75
# maximum_amplitude_scale_factor
amplitude_scale_max = 1.5
# number of samples to take before an event timestep
samples_before = 40
# number of samples to take after an event timestep
samples_after = 40
# threshold a channel must exceed to be considered part of an event
event_threshold = -30
# point in the raw file at which the data starts
offset = 0
# whether or not to copy the source to the target
copy = False
# whether or not to overwrite a target file if it already exists
overwrite = True
# whether or not to subtract out the true units from the hybrid data (EXPERIMENTAL)
subtract = False
