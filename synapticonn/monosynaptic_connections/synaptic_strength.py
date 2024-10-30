""" synaptic_strength.py

Modules for validating synaptic strength.
"""

import warnings
import numpy as np

from joblib import Parallel, delayed

from synapticonn.postprocessing.crosscorrelograms import compute_crosscorrelogram_dual_spiketrains
from synapticonn.utils.errors import SpikeTimesError


##########################################################
##########################################################


def calculate_synaptic_strength(pre_spike_train=None, post_spike_train=None,
                                jitter_range_ms=10, num_iterations=1000,
                                max_lag_ms=25, bin_size_ms=0.5,
                                half_window_ms=5, n_jobs=-1):
    """ Calculate the synaptic strength between two spike trains.

    Parameters
    ----------
    pre_spike_train : np.ndarray
        The spike times for the pre-synaptic cell.
    post_spike_train : np.ndarray
        The spike times for the post-synaptic cell.
    jitter_range_ms : float
        Jin the range (in milliseconds) within which to jitter each spike time.
        Each spike will be shifted by a random amount in the range [-jitter_range_ms, +jitter_range_ms].
        Default is 10 ms.
    num_iterations : int
        The number of jittered cross-correlograms to compute.
        Default is 1000.
    max_lag_ms : float
        The maximum lag to compute the cross-correlogram (in milliseconds).
        Default is 25 ms.
    bin_size_ms : float, optional
        The size of each bin in the cross-correlogram (in milliseconds).
        Default is 0.5 ms.
    half_window_ms : float, optional
        The half-width of the window around the zero-lag time (in milliseconds).
        Default is 5 ms.
    n_jobs : int, optional
        The number of jobs to run in parallel. Default is -1.

    Returns
    -------
    synaptic_strength_data : dict
        Dictionary containing the original cross-correlogram counts,
        the jittered cross-correlogram counts, the synaptic strength value,
        and the confidence intervals.

    CCG notes
    ---------
    In CCG analysis, if one unit fires before the other, the peak of the CCG
    will be positive. This peak will be centred around the zero-lag time.
    This delay is typically 1-5 ms for excitatory synapses, but can vary
    depending on the type of synapse and the distance between the cells.
    True monosynaptic connections will have a peak at zero lag.

    If the time-lag is less than zero, this indicates that the post-synaptic
    cell is firing before the pre-synaptic cell. In such cases, the direction
    of causality could be reversed, and change depending on which cell is
    considered the pre-synaptic cell. This is why it is important to consider
    the direction of causality when interpreting the CCG results.
    This bidirectional can help confirm the presence of a monosynaptic
    connections.

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

    if (pre_spike_train is None) or (post_spike_train is None):
        raise SpikeTimesError("Pre- and post-synaptic spike trains are required.")

    if bin_size_ms > 0.5:
        warnings.warn("Bin size is greater than 0.5 ms. This may affect the accuracy of the synaptic strength value.", UserWarning)
    if max_lag_ms > 25:
        warnings.warn("Maximum lag is greater than 25 ms. It is recommended to calculate synaptic strength within a 25 ms range.", UserWarning)
    if jitter_range_ms > 10:
        warnings.warn("Jitter range is greater than 10 ms. This may affect the accuracy of the synaptic strength value.", UserWarning)
    if num_iterations < 1000:
        warnings.warn("Number of iterations is less than 1000. This may affect the accuracy of the synaptic strength value.", RuntimeWarning)

    #########################

    synaptic_strength_data = _return_jittered_ccg(pre_spike_train, post_spike_train,
                                                  num_iterations, max_lag_ms,
                                                  bin_size_ms, jitter_range_ms, n_jobs)

    synaptic_strength_data.update(
        _return_synaptic_strength_zscore(synaptic_strength_data['ccg_bins'],
                                         synaptic_strength_data['original_ccg_counts'],
                                         synaptic_strength_data['jittered_ccg_counts'],
                                         half_window_ms, bin_size_ms))

    return synaptic_strength_data


def _return_synaptic_strength_zscore(ccg_bins, original_ccg_counts,
                                     jittered_ccg_counts, half_window_ms=5,
                                     bin_size_ms=0.5):
    """ Calculate the synaptic strength as the Z-score of the peak bin
    count within a specified window in the original CCG.

    Parameters
    ----------
    ccg_bins : np.ndarray
        The time bins for the cross-correlogram.
    original_ccg_counts : np.ndarray
        The original cross-correlogram counts.
    jittered_ccg_counts : np.ndarray
        The jittered cross-correlogram counts.
    half_window_ms : float, optional
        The half-width of the window around the zero-lag time (in milliseconds).
        Default is 5 ms.
    bin_size_ms : float, optional
        The size of each bin in the cross-correlogram (in milliseconds).
        Default is 0.5 ms.

    Returns
    -------
    synaptic_strength_zscore : dict
        Dictionary containing the synaptic strength value, and the confidence intervals.

    References
    ----------
    [1] STAR Protoc. 2024 Jun 21;5(2):103035. doi: 10.1016/j.xpro.2024.103035. Epub 2024 Apr 27
    """

    assert len(original_ccg_counts) == (jittered_ccg_counts.shape[1]), "Original and jittered CCG counts must have the same length."

    # define the window around the zero-lag time
    window_bins = int((half_window_ms*2) / (2 * bin_size_ms))
    mid_bin = len(ccg_bins) // 2  # the center bin corresponds to zero lag
    window_slice = slice(mid_bin - window_bins, mid_bin + window_bins + 1)  # slice the window

    # identify the peak bin count within window
    x_real = np.max(original_ccg_counts[window_slice])

    # compute mean and standard deviation of the jittered CCGs
    jittered_window_counts = jittered_ccg_counts[:, window_slice]
    m_jitter = np.mean(jittered_window_counts)
    s_jitter = np.std(jittered_window_counts)

    # calculate Z-score
    if s_jitter > 0:
        synaptic_strength = (x_real - m_jitter) / s_jitter
    else:
        synaptic_strength = np.inf  # if no variance, Z is undefined or infinite

    # calculate confidence intervals
    high_ci = np.percentile(jittered_window_counts, 99, axis=0)
    low_ci = np.percentile(jittered_window_counts, 1, axis=0)

    synaptic_strength_zscore = {'synaptic_strength': synaptic_strength, 'high_ci': high_ci, 'low_ci': low_ci}

    return synaptic_strength_zscore


def _return_jittered_ccg(pre_spike_train, post_spike_train, num_iterations=1000,
                         max_lag_ms=25, bin_size_ms=0.5, jitter_range_ms=10, n_jobs=-1):
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
    n_jobs : int, optional
        The number of jobs to run in parallel. Default is -1.
        If -1, all CPUs are used. If 1, no parallel computing code is used at all,
        which is useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.

    Returns
    -------
    jittered_ccg_data : dict
        Dictionary containing the original cross-correlogram counts and the jittered cross-correlogram counts.

    Notes
    -----
    For reproducibility, a seed is recommended to be set for each iteration.
    This is to ensure that the same jittered spike train is generated for each iteration.
    Hence, the synaptic strength value is consistent across multiple runs.
    """

    assert num_iterations > 2, "Number of iterations must be greater than zero."

    original_ccg_counts, ccg_bins = compute_crosscorrelogram_dual_spiketrains(pre_spike_train, post_spike_train, bin_size_ms, max_lag_ms)

    # jitter a single spike train across multiple iterations
    # note :: a seed is applied to each iteration for reproducibility
    def single_jitter_iteration(seed):
        jittered_post_spike_train = _apply_jitter(post_spike_train, jitter_range_ms, seed=seed)
        jittered_ccg_counts, _ = compute_crosscorrelogram_dual_spiketrains(pre_spike_train, jittered_post_spike_train, bin_size_ms, max_lag_ms)
        return jittered_ccg_counts

    jittered_ccgs_list = Parallel(n_jobs=n_jobs)(delayed(single_jitter_iteration)(i) for i in range(num_iterations))
    jittered_ccgs = np.vstack(jittered_ccgs_list)

    jittered_ccg_data = {'ccg_bins': ccg_bins, 'original_ccg_counts': original_ccg_counts, 'jittered_ccg_counts': jittered_ccgs}

    return jittered_ccg_data


def _apply_jitter(spike_train, jitter_range_ms, seed=None):
    """ Apply random jitter to a spike train within a specified range.

    This is an internal function used to apply random jitter to a spike train.
    It is not recommended to use this function directly.

    Parameters
    ----------
    spike_train : array_like
        The original spike times for a single cell.
    jitter_range_ms : float
        The range (in milliseconds) within which to jitter each spike time.
        Each spike will be shifted by a random amount in the range [-jitter_range_ms, +jitter_range_ms].
    seed : int, optional
        Random seed for reproducibility. Default is 0.

    Returns
    -------
    jittered_spike_train : np.ndarray
        The spike train with added random jitter.

    Notes
    -----
    The output spike train is sorted in ascending order.
    This is to ensure that the jittered spike times are in
    the correct temporal order.

    A seed is recommended to be set for each iteration to ensure
    reproducibility.

    References
    ----------
    [1] https://numpy.org/doc/2.0/reference/random/generated/numpy.random.seed.html 
    """

    assert jitter_range_ms > 0, "Jitter range must be greater than zero."

    if seed is not None:
        np.random.seed(seed)  # add seed for reproducibility

    jitter = np.random.uniform(-jitter_range_ms, jitter_range_ms, size=len(spike_train))
    jittered_spike_train = spike_train + jitter

    sorted_jittered_spike_train = np.sort(jittered_spike_train)  # sort to ensure temporal order

    return sorted_jittered_spike_train
