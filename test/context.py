# Copyright (C) 2018 Vidrio Technologies. All rights reserved.

import os
from os import path as op

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from dotenv import load_dotenv
load_dotenv()
testbase = op.abspath(os.getenv("TESTBASE"))

import pytest

import hybridfactory
