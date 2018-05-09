import numpy as np

# required parameters

raw_source_file = r"C:\Users\Alan\Documents\Data\eMouse\sim_binary.dat"
raw_target_file = r"C:\Users\Alan\Documents\Data\eMouse\sim_binary_GT.dat"
data_type = np.int16
sample_rate = 25000
output_type = "phy"
data_directory = r"C:\Users\Alan\Documents\Data\eMouse"
probe_type = "eMouse"
ground_truth_units = [8, 16, 18, 26, 32, 34]

# optional parameters

random_seed = 10191
generator_type = "steinmetz"
num_singular_values = 6
channel_shift = 4
time_jitter = 500
samples_before = 30
samples_after = 30
event_threshold = -30
offset = 0
extra_channels = 0
copy = True
overwrite = True
