""" correlogram_utils.py

Utilities for correlogram generation.
"""

import numpy as np


##########################################################
##########################################################


def make_bins(max_lag_t, bin_size_t):
    """ Make bins for correlograms.

    Parameters
    ----------
    max_lag_t : float
        Maximum lag to compute the correlograms (in milliseconds).
    bin_size_t : float
        Bin size of the correlograms (in milliseconds).

    Returns
    -------
    bins : array-like
        Bin edges for the correlograms.
    """

    num_bins = int(2 * max_lag_t / bin_size_t) + 1
    bins = np.linspace(-max_lag_t, max_lag_t, num_bins)

    assert len(bins) > 1, "Not enough bins created. Increase max_lag_t or decrease bin_size_t."

    return bins