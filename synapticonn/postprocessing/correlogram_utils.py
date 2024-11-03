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

    num_bins = int(2 * max_lag_ms / bin_size_ms) + 1
    bins = np.linspace(-max_lag_ms, max_lag_ms, num_bins)

    assert len(bins) > 1, "Not enough bins created. Increase max_lag_ms or decrease bin_size_ms."

    return bins