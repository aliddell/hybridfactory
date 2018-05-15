"""Copyright (C) 2018 Vidrio Technologies. All rights reserved."""

import numpy as np
import matplotlib.pyplot as plt


def plot_match(template, response):
    template_diff = np.abs(template - response)

    curve = np.log10(np.sort(template_diff[template_diff > 0]))

    # find the "inflection point" of this curve and use it as a tolerance
    tolexp = curve[np.abs(curve - np.mean(curve)).argmin()]
    tol = 10 ** tolexp

    # construct a bounding box around the event
    rowmax = np.abs(template_diff).max(axis=1).argmax()
    colmax = np.abs(template_diff).max(axis=0).argmax()

    # traverse the four cardinal directions until you come to an edge or something close to zero
    top_bound = bottom_bound = rowmax
    left_bound = right_bound = colmax

    while top_bound > 0 and template_diff[top_bound, colmax] > tol:
        top_bound -= 1
    while bottom_bound < template_diff.shape[0] and template_diff[bottom_bound, colmax] > tol:
        bottom_bound += 1

    while left_bound > 0 and template_diff[rowmax, left_bound] > tol:
        left_bound -= 1
    while right_bound < template_diff.shape[1] and template_diff[rowmax, right_bound] > tol:
        right_bound += 1

    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharey=True)
    ax0.imshow(template[top_bound:bottom_bound, left_bound:right_bound].T, cmap="coolwarm",
               extent=(left_bound, right_bound, bottom_bound, top_bound), aspect="auto")
    fig.colorbar(ax0.images[0], orientation="vertical", ax=ax0)

    ax1.imshow(response[top_bound:bottom_bound, left_bound:right_bound].T, cmap="coolwarm",
               extent=(left_bound, right_bound, bottom_bound, top_bound), aspect="auto")
    fig.colorbar(ax1.images[0], orientation="vertical", ax=ax1)

    return fig, (ax0, ax1)
