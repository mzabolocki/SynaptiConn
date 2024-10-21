"""
acg.py

Modules for plotting autocorrelograms.
"""

import numpy as np
import matplotlib.pyplot as plt

from synapticonn.utils.mod_utils import check_dependency


##########################################################
##########################################################

@check_dependency(plt, 'matplotlib')
def plot_acg(acg, bin_size, t_stop, ax=None, **kwargs):
    """Plot an autocorrelogram.

    Parameters
    ----------
    acg : array-like
        Autocorrelogram to plot.
    bin_size : float
        Bin size of the autocorrelogram.
    t_stop : float
        Maximum time to plot.
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

    time = np.arange(-t_stop, t_stop + bin_size, bin_size)
    ax.bar(time, acg, width=bin_size, **kwargs)

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Count')

    return ax


def _compute_autocorrelogram(spike_train, bin_size_ms=1, max_lag_ms=100):
    """ Compute the autocorrelogram of a spike train.

    Parameters
    ----------
    spike_train : array-like
        Spike times (in seconds).
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

    # convert max_lag and bin_size to bins
    max_lag_bins = int(max_lag_ms / bin_size_ms)

    # compute the differences between all spike times
    spike_diffs = np.subtract.outer(spike_train, spike_train)

    # only keep differences within the maximum lag
    spike_diffs = spike_diffs[np.abs(spike_diffs) <= max_lag_ms]

    # compute histogram (binning the differences)
    autocorr, bin_edges = np.histogram(spike_diffs, bins=np.arange(-max_lag_ms, max_lag_ms + bin_size_ms, bin_size_ms))

    # remove the zero-lag bin (since it's the same spike)
    zero_bin = int(len(autocorr) // 2)
    autocorr[zero_bin] = 0

    # compute lag values
    lags = bin_edges[:-1] + bin_size_ms / 2

    return lags, autocorr
