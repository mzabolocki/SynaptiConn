""" presence_ratio.py.

Modules for computing presence ratios.
"""

import warnings
import math

import numpy as np

from synapticonn.utils.errors import RecordingLengthError


####################################################
####################################################


def compute_presence_ratio(spike_train_ms, recording_length_ms,
                           bin_duration_ms=60000, mean_fr_ratio_thresh=0.0, srate=None):
    """ Compute the presence ratio of a spike train.

    Parameters
    ----------
    spike_train_ms : numpy.ndarray
        Spike train in milliseconds.
    recording_length_ms : float
        Length of the recording in milliseconds.
    bin_duration_ms : float
        Duration of each bin in milliseconds.
        By default, this is set to 60 seconds.
    mean_fr_ratio_thresh : float
        Minimum mean firing rate ratio threshold.
        This is the minimum mean firing rate that must be present in a bin
        for the unit to be considered "present" in that bin.
        By default, this is set to 0.0. This means that the unit must have
        at least one spike in each bin to be considered "present."
    srate : float
        Sampling rate in Hz.

    Returns
    -------
    presence_ratio : dict
        Dictionary containing the presence ratio.

    Notes
    -----
    Presence ratio is not a standard metric in the field,
    but it's straightforward to calculate and is an easy way to
    identify incomplete units. It measures the fraction of time during a
    session in which a unit is spiking, and ranges from 0 to
    0.99 (an off-by-one error in the calculation ensures
    that it will never reach 1.0).

    Code is adapted from Spike Interface
    (https://github.com/SpikeInterface/spikeinterface/blob/main/src/spikeinterface/qualitymetrics/misc_metrics.py#L1147)

    References
    ----------
    [1] https://spikeinterface.readthedocs.io/en/stable/modules/qualitymetrics/presence_ratio.html
    [2] https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_quality_metrics.html#ISI-violations 
    """

    assert srate is not None, "The sampling rate must be provided."

    mean_fr_ratio_thresh = float(mean_fr_ratio_thresh)
    if mean_fr_ratio_thresh < 0:
        raise ValueError(f"The mean firing rate ratio threshold must be greater than or equal to 0., not {mean_fr_ratio_thresh}.")
    if mean_fr_ratio_thresh > 1:
        warnings.warn("A mean firing rate ratio threshold > 1.0 may lead to low presence ratios.")

    if recording_length_ms < bin_duration_ms:
        raise ValueError(f"The recording length of {recording_length_ms} "
                        f"ms is shorter than the bin duration of {bin_duration_ms} ms.")

    if bin_duration_ms < 60000:
        warnings.warn("The bin duration is less than 60 seconds. This may lead to inaccurate presence ratios.")

    spike_train_sec = spike_train_ms / 1000  # Ensure spike_train_ms is a NumPy array
    bin_duration_sec = bin_duration_ms / 1000
    recording_length_sec = recording_length_ms / 1000

    if recording_length_sec <= 0:
        raise ValueError("Recording length must be greater than 0.")

    # calculate the number of spikes that must be present in a bin to be considered "present"
    unit_fr = len(spike_train_ms) / recording_length_sec  # Ensure len() is appropriate for spike_train_ms
    bin_n_spikes_thresh = math.floor(unit_fr * bin_duration_sec * mean_fr_ratio_thresh)

    # calculate the number of bins
    bin_duration_samples = int(bin_duration_sec * srate)
    total_length = int(recording_length_sec * srate)
    num_bin_edges = total_length // bin_duration_samples + 1

    # calculate the presence ratio
    bin_edges_in_samples = np.arange(num_bin_edges) * bin_duration_samples
    spike_train_in_samples = spike_train_sec * srate
    h, _ = np.histogram(spike_train_in_samples, bins=bin_edges_in_samples)
    presence_ratio = np.sum(h > bin_n_spikes_thresh) / (len(bin_edges_in_samples) - 1)

    return {"presence_ratio": presence_ratio}