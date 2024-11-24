""" presence_ratio.py.

Modules for computing presence ratios.
"""

import warnings
import math

import numpy as np

from synapticonn.utils.errors import RecordingLengthError


####################################################
####################################################


def compute_presence_ratio(spike_train,
                           recording_length_t,
                           time_unit,
                           bin_duration_sec=60,
                           mean_fr_ratio_thresh=0.0,
                           srate=None):
    """ Compute the presence ratio of a spike train.

    Parameters
    ----------
    spike_train : numpy.ndarray
        Spike train in milliseconds. If in seconds,
        convert to milliseconds.
    recording_length_t : float
        Length of the recording.
    time_unit : str
        Time unit of the recording.
        Options are 'ms' or 's'.
    bin_duration_sec : float
        Duration of each bin in seconds.
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

    # force time unit to seconds
    if time_unit == "s":
        spike_train_sec = spike_train
        recording_length_sec = recording_length_t

    elif time_unit == "ms":
        warnings.warn("Converting spike train and recording length to seconds "
                      "for presence ratio calculation.")
        spike_train_sec = spike_train / 1000
        recording_length_sec = recording_length_t / 1000

    # force float conversion
    mean_fr_ratio_thresh = float(mean_fr_ratio_thresh)

    # validate the input parameters
    _validate_presence_ratio_params(srate, recording_length_sec, bin_duration_sec, mean_fr_ratio_thresh)

    # spike time for unit, in samples
    spike_train_samples = spike_train_sec * srate

    # calculate bin edges
    bin_duration_samples = int((bin_duration_sec * srate))
    total_length = int(recording_length_sec * srate)
    num_bin_edges = total_length // bin_duration_samples + 1
    bin_edges = np.arange(num_bin_edges) * bin_duration_samples

    # calculate thresholds
    unit_fr = len(spike_train_sec) / recording_length_sec
    bin_n_spikes_thresh = math.floor(unit_fr * bin_duration_sec * mean_fr_ratio_thresh)

    # calculate bin edges
    if bin_edges is not None:
        bins = bin_edges
        num_bin_edges = len(bin_edges)
    else:
        bins = num_bin_edges

    h, _ = np.histogram(spike_train_samples, bins=bins)

    presence_ratio = np.sum(h > bin_n_spikes_thresh) / (len(bin_edges) - 1)

    return {"presence_ratio": presence_ratio}


def _validate_presence_ratio_params(srate,
                                    recording_length_t,
                                    bin_duration_sec,
                                    mean_fr_ratio_thresh):
    """ Validate input parameters for presence ratio calculation.

    Parameters
    ----------
    srate : float
        Sampling rate in Hz.
    recording_length_t : float
        Length of the recording in milliseconds.
    bin_duration_sec : float
        Duration of each bin in seconds.
    mean_fr_ratio_thresh : float
        Minimum mean firing rate ratio threshold.
    """

    assert srate > 0, "Sampling rate must be a positive number."
    assert recording_length_t > 0, "Recording length must be greater than 0."
    assert bin_duration_sec > 0, "Bin duration must be greater than 0."

    # --- validate the mean firing rate ratio threshold ---

    if mean_fr_ratio_thresh < 0:
        raise ValueError("The mean firing rate ratio "
                         "threshold must be >= 0.")
    if mean_fr_ratio_thresh > 1:
        warnings.warn("A mean firing rate ratio threshold > 1 may "
                      "lead to low presence ratios.")
    if recording_length_t < bin_duration_sec:
        warnings.warn("Bin duration exceeds recording length. "
                      "Presence ratio will be inaccurate.")

    # --- validate the recording length ---

    # ensure the recording length is greater than the bin duration
    if recording_length_t < bin_duration_sec:
        raise ValueError(f"The recording length of {recording_length_t} "
                         f"sec is shorter than the bin duration of {bin_duration_sec} sec.")

    # --- validate the bin duration ---

    # ensure the bin duration is at least 60 seconds
    if bin_duration_sec < 60:
        warnings.warn("The bin duration is less than 60 seconds."
                      "This may lead to inaccurate presence ratios.")
