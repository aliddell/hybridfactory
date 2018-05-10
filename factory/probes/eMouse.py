"""Copyright (C) 2018 Vidrio Technologies. All rights reserved."""

import numpy as np

"""
See https://github.com/cortex-lab/KiloSort/tree/master/eMouse for details.
"""

NCHANS = 34

# the full channel map for the eMouse array
channel_map = np.array([32, 33, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
                        26, 28, 30, 0, 1, 2, 3, 4, 5], dtype=np.int64)

# reference channels
refchans = np.array([32, 33])
connected = ~np.isin(channel_map, refchans)

# physical location of each channel on the probe
xcoords = 20 * np.array([np.nan, np.nan, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                         1, 0, 1, 0, 1, 0])
ycoords = 20 * np.array([np.nan, np.nan, 7, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 17, 17, 18, 18,
                         19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24])
channel_positions = np.hstack((xcoords[:, np.newaxis], ycoords[:, np.newaxis]))  # NCHANS x 2

# default amount to shift by
default_shift = 4
