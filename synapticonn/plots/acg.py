"""
acg.py

Modules for plotting autocorrelograms.
"""

import pathlib

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import style

from synapticonn.utils.mod_utils import check_dependency
from synapticonn.plots.checks import check_empty, check_ndim


##########################################################
##########################################################


style.use(pathlib.Path('synapticonn', 'plots', 'plots.mplstyle'))


##########################################################
##########################################################

@check_dependency(plt, 'matplotlib')
@check_ndim
@check_empty
def plot_acg(spike_train_ms, bin_size_ms=1, max_lag_ms=100, ax=None, **kwargs):
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
    **kwargs
        Additional keyword arguments passed to `ax.bar`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axis with the autocorrelogram plot.
    """
    if ax is None:
        ax = plt.gca()

    lags, autocorr = _compute_autocorrelogram(spike_train_ms, bin_size_ms, max_lag_ms)

    ax.bar(lags, autocorr, width=bin_size_ms, **kwargs)
    ax.set_xlabel('Time lag (ms)')
    ax.set_ylabel('Spike counts/bin')

    return ax


def _compute_autocorrelogram(spike_train_ms, bin_size_ms=1, max_lag_ms=100):
    """ Compute the autocorrelogram of a spike train.

    Parameters
    ----------
    spike_train_ms : array-like
        Spike times (in milliseconds).
    bin_size_ms : float, optional
        Bin size of the autocorrelogram (in milliseconds).
    max_lag_ms : float, optional
        Maximum lag to compute the autocorrelogram (in milliseconds).

    Returns
    -------
    lags : array-like
        Lag values (in milliseconds).
    autocorr : array-like
        Autocorrelogram values.
    """

    # compute the differences between all spike times (in ms)
    spike_diffs = np.subtract.outer(spike_train_ms, spike_train_ms)

    # keep only differences within the maximum lag
    spike_diffs = spike_diffs[np.abs(spike_diffs) <= max_lag_ms]

    # compute histogram (binning the differences with given bin size)
    bins = np.arange(-max_lag_ms, max_lag_ms + bin_size_ms, bin_size_ms)
    autocorr, bin_edges = np.histogram(spike_diffs, bins=bins)

    # remove the zero-lag bin (since it's the same spike)
    zero_bin = len(autocorr) // 2
    autocorr[zero_bin] = 0

    # compute lag values centered on each bin
    lags = bin_edges[:-1] + bin_size_ms / 2

    return lags, autocorr
