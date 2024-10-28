""" synaptic_strength.py

Modules for validating synaptic strength.
"""

import numpy as np

from synapticonn.postprocessing.crosscorrelograms import compute_crosscorrelogram_dual_spiketrains
from synapticonn.postprocessing.correlogram_utils import make_bins


##########################################################
##########################################################


def calculate_jittered_ccg(pre_spike_train, post_spike_train, num_iterations=1000,
                           max_lag_ms=25, bin_size_ms=0.5, jitter_range_ms=10):
    """ Calculate the jittered cross-correlogram and confidence intervals.

    Parameters
    ----------
    pre_spike_train : np.ndarray
        The spike times for the pre-synaptic cell.
    post_spike_train : np.ndarray
        The spike times for the post-synaptic cell.
    num_iterations : int
        The number of jittered cross-correlograms to compute.
        Default is 1000.
    max_lag_ms : float
        The maximum lag to compute the cross-correlogram (in milliseconds).
        Default is 25 ms.
    bin_size_ms : float, optional
        The size of each bin in the cross-correlogram (in milliseconds).
        Default is 0.5 ms.

    Returns
    -------
    ccg_bins : np.ndarray
        The bin edges for the cross-correlogram.
    ccg_counts : np.ndarray
        The counts for the actual cross-correlogram.
    jittered_ccgs : np.ndarray
        The counts for the jittered cross-correlograms.

    References
    ----------
    [1] STAR Protoc. 2024 Jun 21;5(2):103035. doi: 10.1016/j.xpro.2024.103035. Epub 2024 Apr 27
    """

    # compute cross-correlogram for dual spike trains
    ccg_counts, ccg_bins = compute_crosscorrelogram_dual_spiketrains(pre_spike_train, post_spike_train, max_lag_ms, bin_size_ms)

    # jitter a single spike train across multiple iterations
    jittered_ccgs = np.zeros((num_iterations, len(ccg_bins)))
    for i in range(num_iterations):
        jittered_post_spike_train = apply_jitter(post_spike_train, jitter_range_ms)
        _, jittered_ccg_counts = compute_crosscorrelogram_dual_spiketrains(pre_spike_train, jittered_post_spike_train, max_lag_ms, bin_size_ms)
        jittered_ccgs[i, :] = jittered_ccg_counts

    return ccg_bins, ccg_counts, jittered_ccgs


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

    Notes
    -----
    The output spike train is sorted in ascending order.
    This is to ensure that the jittered spike times are in
    the correct temporal order.
    """

    # generate random jitter values for each spike
    jitter = np.random.uniform(-jitter_range_ms, jitter_range_ms, size=len(spike_train))
    jittered_spike_train = spike_train + jitter

    # sort the jittered spike train
    sorted_jittered_spike_train = np.sort(jittered_spike_train)

    return sorted_jittered_spike_train


#############################
#############################


# TO DO :: add back in 
# # CI (1% and 99%)
# upper_conf = np.percentile(jittered_ccgs, 99, axis=0)
# lower_conf = np.percentile(jittered_ccgs, 1, axis=0)
    

# def compute_synaptic_strength(ccg_actual, jittered_ccgs, window_ms, bin_size_ms):
#     """Compute the standardized value (Z) to assess the strength of the synaptic interaction.

#     Parameters
#     ----------
#     ccg_actual : array_like
#         The cross-correlogram of the actual spike trains.
#     jittered_ccgs : array_like
#         2D array where each row is a jittered cross-correlogram (shape: num_repeats x num_bins).
#     window_ms : float
#         The window in milliseconds around zero within which to find the peak bin count.
#     bin_size_ms : float
#         The size of each bin in milliseconds.

#     Returns
#     -------
#     float
#         The standardized Z-value representing the strength of the synaptic interaction.
#     """

#     # Calculate the indices corresponding to the desired window around zero
#     half_window_bins = int(window_ms / (2 * bin_size_ms))
#     mid_bin = len(ccg_actual) // 2  # The center bin corresponds to zero lag
#     window_slice = slice(mid_bin - half_window_bins, mid_bin + half_window_bins + 1)

#     # Identify the peak bin count within the window in the actual CCG
#     x_real = np.max(ccg_actual[window_slice])

#     # Compute mean and standard deviation of the jittered CCGs within the same window
#     jittered_window_counts = jittered_ccgs[:, window_slice]
#     m_jitter = np.mean(jittered_window_counts)
#     s_jitter = np.std(jittered_window_counts)

#     # Calculate the Z-score
#     if s_jitter > 0:
#         Z = (x_real - m_jitter) / s_jitter
#     else:
#         Z = np.inf  # If no variance in jittered counts, Z is undefined or infinite

#     return Z