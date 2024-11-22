""" autocorrelograms.py

Modules for generating autocorrelograms.
"""

import numpy as np

from synapticonn.postprocessing.correlogram_utils import make_bins

##########################################################
##########################################################


def compute_autocorrelogram(spike_train_ms, bin_size_t=1, max_lag_t=100):
    """ Compute the autocorrelogram of a spike train.

    Parameters
    ----------
    spike_train_ms : array-like
        Spike times (in milliseconds).
    bin_size_t : float, optional
        Bin size of the autocorrelogram (in milliseconds).
    max_lag_t : float, optional
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
    spike_diffs = spike_diffs[np.abs(spike_diffs) <= max_lag_t]

    # compute histogram (binning the differences with given bin size)
    bins = make_bins(max_lag_t, bin_size_t)
    autocorr, bin_edges = np.histogram(spike_diffs, bins=bins)

    # remove the zero-lag bin (since it's the same spike)
    zero_bin = len(autocorr) // 2
    autocorr[zero_bin] = 0

    # compute lag values centered on each bin
    lags = bin_edges[:-1] + bin_size_t / 2

    return lags, autocorr
