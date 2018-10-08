# Copyright (C) 2018 Vidrio Technologies. All rights reserved.

import os
from os import path as op
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from dotenv import load_dotenv
load_dotenv()


def md5sum(filename):
    # hat tip to this guy: https://stackoverflow.com/questions/22058048/hashing-a-file-in-python#22058673
    chunk_size = 65536  # read in 64 KiB chunks

    result = hashlib.md5()

    with open(filename, "rb") as fh:
        while True:
            data = fh.read(chunk_size)
            if not data:
                break
            result.update(data)

    return result.hexdigest()

testbase = op.abspath(os.getenv("TESTBASE"))
