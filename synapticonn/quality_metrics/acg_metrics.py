""" acg_metrics.py.

Modules for computing and plotting autocorrelogram metrics.
"""


import numpy as np
from scipy.stats import zscore


####################################################
####################################################


def isi_violations(spike_train_ms, min_isi, reference_duration): 
    """ Interspike-interval (ISI) violations."""

    isi_violations = np.diff(spike_train_ms)
    num_spikes = len(spike_train_ms)

    num_violations = sum(isi_violations < reference_duration)
    violation_time = 2 * num_spikes * (reference_duration - min_isi)

    total_rate = num_spikes / spike_train_ms[-1]

    violation_rate = num_violations / violation_time

    fprate = violation_rate / total_rate

    return fprate, num_violations


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

    