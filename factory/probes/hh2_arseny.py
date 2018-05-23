# Copyright (C) 2018 Vidrio Technologies. All rights reserved.

import numpy as np

"""
See https://github.com/JaneliaSciComp/JRCLUST/blob/master/prb/hh2_arseny.prb for details.
"""

NCHANS = 256

# the full channel map for the Arseny probe
channel_map = np.hstack((np.arange(64), np.array([111, 110, 109, 108, 106, 107, 104, 105, 102, 103, 100, 101, 98, 99,
                                                  96, 97, 80, 81, 82, 83, 85, 84, 87, 86, 89, 88, 91, 90, 93, 92, 95,
                                                  94, 65, 64, 67, 66, 69, 68, 71, 70, 73, 72, 75, 74, 76, 77, 78, 79,
                                                  126, 127, 124, 125, 122, 123, 120, 121, 118, 119, 116, 117, 115, 114,
                                                  113, 112]), np.arange(128, 256)))

# reference channels
refchans = np.hstack((np.arange(64), [99], np.arange(128, 256)))
connected = ~np.isin(channel_map, refchans)

# physical location of each channel on the probe
xcoords = np.hstack((np.repeat(np.nan, NCHANS//4), np.repeat([0, 250], NCHANS//8), np.repeat(np.nan, NCHANS//2)))
ycoords = np.hstack((np.repeat(np.nan, NCHANS//4), np.tile(25*np.arange(32), 2), np.repeat(np.nan, NCHANS//2)))
channel_positions = np.hstack((xcoords[:, np.newaxis], ycoords[:, np.newaxis]))  # NCHANS x 2

# default amount to shift by
default_shift = 4
