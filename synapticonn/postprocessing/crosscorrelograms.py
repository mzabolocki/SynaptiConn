"""
crosscorrelograms.py

Modules for generating crosscorrelograms.
"""

import numpy as np


##########################################################
##########################################################


def compute_cross_correlogram(spike_train_set, labels, bin_size_ms, max_lag_ms):
    """ Compute the cross-correlogram between all pairs of spike trains in a set.

    Entry function to compute correlograms across all units in a 'spike_train_set'.
    This is a dictionary, indexed by unit ID and containing spike times (in milliseconds).

    Parameters
    ----------
    spike_train_set : dict
        Dictionary containing spike times for each unit.
        Indexed by unit ID.
        Spike times are in milliseconds.
    labels : list
        List of labels for each unit.
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

    assert len(spike_train_set) > 1, 'The spike train set must contain at least two spike trains.'

    cross_corr_dict = {}
    bins_dict = {}
    for i, spike_train_1 in enumerate(spike_train_set):
        for j, spike_train_2 in enumerate(spike_train_set):
            cross_corr, bins = _compute_crosscorrelogram_dual_spiketrains(spike_train_1, spike_train_2, bin_size_ms, max_lag_ms)
            cross_corr_dict[f'{labels[i]}_{labels[j]}'] = cross_corr
            bins_dict[f'{labels[i]}_{labels[j]}'] = bins

    cross_correlograms_data = {'cross_correllations': cross_corr_dict, 'bins': bins_dict}

    return cross_correlograms_data


def _compute_crosscorrelogram_dual_spiketrains(spike_train_1, spike_train_2, bin_size_ms, max_lag_ms):
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

    bins = _make_bins(max_lag_ms, bin_size_ms)
    cross_corr, _ = np.histogram(time_diffs, bins)

    return cross_corr, bins


def _make_bins(max_lag_ms, bin_size_ms):
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
