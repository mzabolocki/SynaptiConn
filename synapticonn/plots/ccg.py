"""
ccg.py

Modules for plotting cross-correlograms.
"""

import pathlib

import numpy as np
import matplotlib.pyplot as plt

from synapticonn.utils.mod_utils import check_dependency
from synapticonn.plots.checks import check_empty, check_ndim
from synapticonn.plots.style import apply_plot_style


##########################################################
##########################################################


style_path = pathlib.Path('synapticonn', 'plots', 'settings.mplstyle')
plt.style.use(style_path)  # set globally


##########################################################
##########################################################


@apply_plot_style(style_path=pathlib.Path('synapticonn', 'plots', 'settings.mplstyle'))
@check_dependency(plt, 'matplotlib')
@check_ndim
@check_empty
def plot_ccg(cross_correlograms_data, bin_size_ms=1, max_lag_ms=100, labels=True, ax=None, **kwargs):
    """Plot an autocorrelogram for a single spike train.

    Parameters
    ----------
    spike_train_ms : array-like
        Spike times (in milliseconds).
    bin_size_ms : float
        Bin size of the autocorrelogram (in milliseconds).
        Default is 1 ms.
    max_lag_ms : float
        Maximum lag to compute the autocorrelogram (in milliseconds).
        Default is 100 ms.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on.
    labels: bool, optional
        Whether to add axis labels. Default
    **kwargs
        Additional keyword arguments passed to `ax.bar`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axis with the autocorrelogram plot.
    """

    if ax is None:
        ax = plt.gca()

    pair_identifiers = list(cross_correlograms_data['cross_correllations'].keys())

    labels = [str(label) for label in labels]
    fig, ax = plt.subplots(len(labels), len(labels), figsize=(25, 15), sharey=False, sharex=True)

    for count, pair_id in enumerate(pair_identifiers):

        cross_corr = cross_correlograms_data['cross_correllations'][pair_id]
        bins = cross_correlograms_data['bins'][pair_id]
        bin_size = bins[1] - bins[0]

        y = np.where(np.array(labels) == pair_id.split('_')[0])[0][0]
        x = np.where(np.array(labels) == pair_id.split('_')[1])[0][0]

        if x == y:
            ax[x, y].bar(bins[:-1], cross_corr, width=bin_size, align='edge', color='green')
        else:
            ax[x, y].bar(bins[:-1], cross_corr, width=bin_size, align='edge', color='blue')

        if x == 0:
            ax[x, y].set_title(labels[y])
        if y == 0:
            ax[x, y].set_ylabel(labels[x])
        if x == len(labels) - 1:
            ax[x, y].set_xlabel('Time lag (ms)')

    return ax
