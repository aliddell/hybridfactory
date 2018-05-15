"""Copyright (C) 2018 Vidrio Technologies. All rights reserved."""

import numpy as np

"""
See https://github.com/cortex-lab/neuropixels/wiki/About_Neuropixels for details.
"""

NCHANS = 385  # 384 channels + sync channel

# the full channel map for the Neuropixels Phase 3A array
channel_map = np.arange(NCHANS)

# reference channels
refchans = np.array([36, 75, 112, 151, 188, 227, 264, 303, 340, 379, 384])
connected = ~np.isin(channel_map, refchans)

# physical location of each channel on the probe
xcoords = np.hstack((np.tile([43, 11, 59, 27], (NCHANS-1)//4), np.nan))  # 43 11 59 27 43 11 59 27 ...
ycoords = 20 * np.hstack((np.repeat(1 + np.arange((NCHANS-1)//2), 2), np.nan))  # 20 20 40 40 ... 3820 3820 3840 3840
channel_positions = np.hstack((xcoords[:, np.newaxis], ycoords[:, np.newaxis]))  # NCHANS x 2

# default amount to shift by
default_shift = 20
