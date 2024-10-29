""" synaptic_strength.py

Modules for validating synaptic strength.
"""

import numpy as np

from synapticonn.postprocessing.crosscorrelograms import compute_crosscorrelogram_dual_spiketrains


##########################################################
##########################################################

def calculate_synaptic_strength(pre_spike_train, post_spike_train, jitter_range_ms=10,
                                num_iterations=1000, max_lag_ms=25, bin_size_ms=0.5,
                                half_window_ms=5, verbose=True):

    # return jittered ccg
    original_ccg_bins, original_ccg_counts, jittered_ccg_counts = _return_jittered_ccg(pre_spike_train,
                                                                                       post_spike_train,
                                                                                       num_iterations=num_iterations,
                                                                                       max_lag_ms=max_lag_ms,
                                                                                       bin_size_ms=bin_size_ms,
                                                                                       jitter_range_ms=jitter_range_ms)

    # identify the number of bins within the window, centered around zero lag
    window_bins = int((half_window_ms*2) / (2 * bin_size_ms))

    # slice the CCG bins within the window
    mid_bin = len(ccg_bins) // 2  # the center bin corresponds to zero lag
    window_slice = slice(mid_bin - window_bins, mid_bin + window_bins + 1)

    # identify the peak bin count within the window in the original CCG
    x_real = np.max(ccg_counts[window_slice])

    # compute mean and standard deviation of the jittered CCGs within the same window
    jittered_window_counts = jittered_ccgs[:, window_slice]
    m_jitter = np.mean(jittered_window_counts)
    s_jitter = np.std(jittered_window_counts)

    # calculate the synaptic stength as the Z-score
    if s_jitter > 0:
        synaptic_strength = (x_real - m_jitter) / s_jitter
    else:
        synaptic_strength = np.inf  # if no variance in jittered counts, Z is undefined or infinite

    if verbose:
        print(f'Window bins: {window_bins} | Window length: {half_window_ms*2} ms')
        print(f'Max time lag for centred window: {ccg_bins[window_slice][0]}')
        print(f'Min time lag for centred window: {ccg_bins[window_slice][-1]}')
        print(f'Synaptic strength: {synaptic_strength}')


def _return_jittered_ccg(pre_spike_train, post_spike_train, num_iterations=1000,
                         max_lag_ms=25, bin_size_ms=0.5, jitter_range_ms=10):
    """ Return the jittered cross-correlogram.

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
    jittered_ccg_data : dict
        Dictionary containing the original cross-correlogram counts and the jittered cross-correlogram counts.

    CCG notes
    ---------
    In CCG analysis, if one unit fires before the other, the peak of the CCG will be positive.
    This peak will be centred around the zero-lag time. This delay is typically 1-5 ms for excitatory
    synapses, but can vary depending on the type of synapse and the distance between the cells.
    True monosynaptic connections will have a peak at zero lag.

    If the time-lag is less than zero, this indicates that the post-synaptic cell is
    firing before the pre-synaptic cell. In such cases, the direction of causality could be reversed,
    and change depending on which cell is considered the pre-synaptic cell. This is why it is
    important to consider the direction of causality when interpreting the CCG results. This bidirectional
    can help confirm the presence of a monosynaptic connections.

    Module notes
    ------------
    A single spike train is jittered across multiple iterations to generate
    a distribution of jittered cross-correlograms. This process is repeated
    across a number of iterations to estimate the confidence intervals [1].
    This is introduced to test for the statistical significance of the actual
    cross-correlogram [1].

    It is recommended that the number of iterations be at least 1000 to
    obtain a reliable estimate of the confidence intervals [1].

    The jitter range is recommended to be within a 10 ms range [1].

    References
    ----------
    [1] STAR Protoc. 2024 Jun 21;5(2):103035. doi: 10.1016/j.xpro.2024.103035. Epub 2024 Apr 27
    """

    # compute cross-correlogram for dual spike trains
    original_ccg_counts, ccg_bins = compute_crosscorrelogram_dual_spiketrains(pre_spike_train, post_spike_train, bin_size_ms, max_lag_ms)

    # jitter a single spike train across multiple iterations
    jittered_ccgs = np.zeros((num_iterations, len(ccg_counts)))
    for i in range(num_iterations):
        jittered_post_spike_train = _apply_jitter(post_spike_train, jitter_range_ms)
        jittered_ccg_counts, jittered_ccg_bins = compute_crosscorrelogram_dual_spiketrains(pre_spike_train, jittered_post_spike_train, bin_size_ms, max_lag_ms)
        jittered_ccgs[i, :] = jittered_ccg_counts

    # note :: ccg bins are the same for both original and jittered ccgs
    jittered_ccg_data = {'ccg_bins': ccg_bins, 'original_ccg_counts': ccg_counts, 'jittered_ccg_counts': jittered_ccgs}

    return jittered_ccg_data


def _apply_jitter(spike_train, jitter_range_ms):
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
    
