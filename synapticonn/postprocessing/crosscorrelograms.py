"""
crosscorrelograms.py

Modules for generating crosscorrelograms.
"""

import numpy as np

from synapticonn.postprocessing.correlogram_utils import make_bins


##########################################################
##########################################################


def compute_crosscorrelogram(spike_times, spike_pairs, bin_size_ms, max_lag_ms):
    """ Compute the cross-correlogram between all pairs of spike trains.

    Parameters
    ----------
    spike_times : dict
        Dictionary containing spike times for each unit. Indexed by unit ID.
    spike_pairs : list
        List of tuples containing the unit IDs of the spike trains to compare.
    bin_size_ms : float
        Bin size of the cross-correlogram (in milliseconds).
    max_lag_ms : float
        Maximum lag to compute the cross-correlogram (in milliseconds).

    Returns
    -------
    cross_correlograms_data : dict
        Dictionary containing cross-correlograms and bins for all pairs
        of spike trains. Indexed by unit ID pairs
    """

    cross_corr_dict = {}
    bins_dict = {}
    for pair in spike_pairs:

        # spk unit labels
        pair_1 = pair[0]
        pair_2 = pair[1]

        # get spike times for each unit
        cross_corr, bins = compute_crosscorrelogram_dual_spiketrains(spike_times[pair_1], spike_times[pair_2], bin_size_ms, max_lag_ms)
        cross_corr_dict[f'{pair_1}_{pair_2}'] = cross_corr
        bins_dict[f'{pair_1}_{pair_2}'] = bins

    crosscorrelogram_data = {'cross_correllations': cross_corr_dict, 'bins': bins_dict}

    return crosscorrelogram_data


def compute_crosscorrelogram_dual_spiketrains(spike_train_1, spike_train_2, bin_size_ms, max_lag_ms):
    """ Compute the cross-correlogram between two spike trains.

    Parameters
    ----------
    spike_train_1 : array_like
        The spike times of the first spike train.
    spike_train_2 : array_like
        The spike times of the second spike train.
    bin_size_ms : float
        The size of the bins in which to bin the time differences (in milliseconds).
    max_lag_ms : float
        The maximum lag to consider (in milliseconds).

    Returns
    -------
    cross_corr : array_like
        The cross-correlogram.
    """

    time_diffs = []
    for spike1 in spike_train_1:
        for spike2 in spike_train_2:
            diff = spike2 - spike1
            if -max_lag_ms <= diff <= max_lag_ms:
                time_diffs.append(diff)

    bins = make_bins(max_lag_ms, bin_size_ms)
    cross_corr, _ = np.histogram(time_diffs, bins)

    return cross_corr, bins
