from context import *

import sys
import hashlib
import pandas as pd

from hybridfactory.utils import cli

modbase = op.join(testbase, "cli")

class TestCLIJRC:
    def setup(self):
        self.testdir = op.join(modbase, "fromjrclust")
        sys.argv = [sys.argv[0]] + ["generate", op.join(self.testdir, "params.py")]
        self.hybrid_out = op.join(self.testdir, "hybrid", "anm420712_20180802_ch0-119bank1_ch120-382bank0_g0_t2.imec.ap.GT.bin")

        cli._main()

    def test_main(self):
        assert(md5sum(self.hybrid_out) == "7f7ec797e81fbbf2e706c4ea2a66c793")
