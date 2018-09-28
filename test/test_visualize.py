from context import *

import matplotlib.pyplot as plt

from hybridfactory.validate import visualize

class TestPlotMatch:
    def setup(self):
        self.templates = np.load(op.join(testbase, "visualize", "templates.npy"))

    def test_plot_match(self):
        fig, (ax0, ax1) = visualize.plot_match(self.templates[:, :, 0], self.templates[:, :, 1])

        assert(isinstance(fig, plt.Figure))
        assert(isinstance(ax0, plt.Axes))
        assert(isinstance(ax1, plt.Axes))

        fig.show()
