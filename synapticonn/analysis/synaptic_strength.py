""" synaptic_strength.py

Modules for validating synaptic strength.
"""

import numpy as np

from synapticonn.postprocessing.correlogram_utils import make_bins


##########################################################
##########################################################


def compute_synaptic_strength(ccg_actual, jittered_ccgs, window_ms, bin_size_ms):
    """Compute the standardized value (Z) to assess the strength of the synaptic interaction.

    Parameters
    ----------
    ccg_actual : array_like
        The cross-correlogram of the actual spike trains.
    jittered_ccgs : array_like
        2D array where each row is a jittered cross-correlogram (shape: num_repeats x num_bins).
    window_ms : float
        The window in milliseconds around zero within which to find the peak bin count.
    bin_size_ms : float
        The size of each bin in milliseconds.

    Returns
    -------
    float
        The standardized Z-value representing the strength of the synaptic interaction.
    """

    # Calculate the indices corresponding to the desired window around zero
    half_window_bins = int(window_ms / (2 * bin_size_ms))
    mid_bin = len(ccg_actual) // 2  # The center bin corresponds to zero lag
    window_slice = slice(mid_bin - half_window_bins, mid_bin + half_window_bins + 1)

    # Identify the peak bin count within the window in the actual CCG
    x_real = np.max(ccg_actual[window_slice])

    # Compute mean and standard deviation of the jittered CCGs within the same window
    jittered_window_counts = jittered_ccgs[:, window_slice]
    m_jitter = np.mean(jittered_window_counts)
    s_jitter = np.std(jittered_window_counts)

    # Calculate the Z-score
    if s_jitter > 0:
        Z = (x_real - m_jitter) / s_jitter
    else:
        Z = np.inf  # If no variance in jittered counts, Z is undefined or infinite

    return Z


def apply_jitter(spike_train, jitter_range_ms):
    """Apply random jitter to a spike train within a specified range.

    Parameters
    ----------
    spike_train : array_like
        The original spike times for a single cell.
    jitter_range_ms : float
        The range (in milliseconds) within which to jitter each spike time.
        Each spike will be shifted by a random amount in the range [-jitter_range_ms, +jitter_range_ms].

    Returns
    -------
    jittered_spike_train : np.ndarray
        The spike train with added random jitter.
    """
    # generate random jitter values for each spike
    jitter = np.random.uniform(-jitter_range_ms, jitter_range_ms, size=len(spike_train))
    jittered_spike_train = spike_train + jitter
    return np.sort(jittered_spike_train)  # sort spike times to maintain temporal order