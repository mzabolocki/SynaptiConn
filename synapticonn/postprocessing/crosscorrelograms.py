"""
crosscorrelograms.py

Modules for generating crosscorrelograms.
"""

import numpy as np
from joblib import Parallel, delayed

from synapticonn.postprocessing.correlogram_utils import make_bins


##########################################################
##########################################################


def compute_crosscorrelogram(spike_times, spike_pairs, bin_size_t, max_lag_t):
    """ Compute the cross-correlogram between all pairs of spike trains.

    Parameters
    ----------
    spike_times : dict
        Dictionary containing spike times for each unit. Indexed by unit ID.
    spike_pairs : list
        List of tuples containing the unit IDs of the spike trains to compare.
    bin_size_t : float
        Bin size of the cross-correlogram.
    max_lag_t : float
        Maximum lag to compute the cross-correlogram.

    Returns
    -------
    cross_correlograms_data : dict
        Dictionary containing cross-correlograms and bins for all pairs
        of spike trains. Indexed by unit ID pairs

    Notes
    -----
    Parallel processing is performed across all available cores.
    """

    def _process_pair(pair):
        pair_1, pair_2 = pair
        cross_corr, bins = compute_crosscorrelogram_dual_spiketrains(
            spike_times[pair_1], spike_times[pair_2], bin_size_t, max_lag_t
        )
        return (pair_1, pair_2), cross_corr, bins

    results = Parallel(n_jobs=-1)(delayed(_process_pair)(pair) for pair in spike_pairs)

    cross_corr_dict = {key: cross_corr for key, cross_corr, _ in results}
    bins_dict = {key: bins for key, _, bins in results}

    crosscorrelogram_data = {'cross_correllations': cross_corr_dict, 'bins': bins_dict}

    return crosscorrelogram_data


def compute_crosscorrelogram_dual_spiketrains(spike_train_1, spike_train_2, bin_size_t, max_lag_t):
    """ Compute the cross-correlogram between two spike trains.

    Parameters
    ----------
    spike_train_1 : array_like
        The spike times of the first spike train.
    spike_train_2 : array_like
        The spike times of the second spike train.
    bin_size_t : float
        The size of the bins in which to bin the time differences (in milliseconds).
    max_lag_t : float
        The maximum lag to consider (in milliseconds).

    Returns
    -------
    cross_corr : array_like
        The cross-correlogram.
    """

    # convert to numpy arrays
    spike_train_1 = np.array(spike_train_1)
    spike_train_2 = np.array(spike_train_2)

    time_diffs = spike_train_2[:, np.newaxis] - spike_train_1

    mask = (time_diffs >= -max_lag_t) & (time_diffs <= max_lag_t)
    valid_diffs = time_diffs[mask]

    bins = make_bins(max_lag_t, bin_size_t)
    cross_corr, _ = np.histogram(valid_diffs, bins)

    return cross_corr, bins

