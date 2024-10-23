""" acg_metrics.py.

Modules for computing and plotting autocorrelogram metrics.
"""


import numpy as np
from scipy.stats import zscore


####################################################
####################################################


def refractory_period_violations(spike_train_ms, bin_size_ms=1, max_lag_ms=100):
    """Compute the number of refractory period violations in an autocorrelogram.

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

    Returns
    -------
    refractory_violations : int
        Number of refractory period violations.
    """

    