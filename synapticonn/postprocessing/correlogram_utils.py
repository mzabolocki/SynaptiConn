""" correlogram_utils.py

Utilities for correlogram generation.
"""

import numpy as np


##########################################################
##########################################################


def make_bins(max_lag_ms, bin_size_ms):
    """ Make bins for correlograms.

    Parameters
    ----------
    max_lag_ms : float
        Maximum lag to compute the correlograms (in milliseconds).
    bin_size_ms : float
        Bin size of the correlograms (in milliseconds).

    Returns
    -------
    bins : array-like
        Bin edges for the correlograms.
    """

    bins = np.arange(-max_lag_ms, max_lag_ms + bin_size_ms, bin_size_ms)

    assert len(bins) >= 1, "No bins created. Increase max_lag_ms or decrease bin_size_ms."

    return bins