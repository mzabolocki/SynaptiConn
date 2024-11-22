""" isi_violations.py.

Modules for computing interspike interval (ISI) violations in a spike train.
"""


import warnings
import numpy as np


####################################################
####################################################


def compute_isi_violations(spike_train_ms, recording_length_t, isi_threshold_ms=1.5, min_isi_ms=0):
    """Compute the number of interspike interval (ISI) violations in a spike train.

    Parameters
    ----------
    spike_train_ms : numpy.ndarray
        Spike train in milliseconds.
    recording_length_t : float
        Length of the recording in milliseconds.
    isi_threshold_ms : float
        Minimum interspike interval in milliseconds.
        This is the minimum time that must elapse between two spikes.
        By default, this is set to 1.5 ms, which is the refractory period of most neurons.
    min_isi_ms : float
        Minimum possible interspike interval in milliseconds.
        This is the artifical refractory period enforced by the
        recording system or the spike sorting algorithm.

    Notes
    -----
    **TO DO** ++ explain the added calculations and the potential biases?

    References
    ----------
    Based on hte metrics orginally implemented in the SpikeInterface package
    (https://github.com/SpikeInterface/spikeinterface/blob/main/src/spikeinterface/qualitymetrics/misc_metrics.py).
    This was based on metrics originally implemented in Ultra Mega Sort [UMS]_.

    This implementation is based on one of the original implementations written in Matlab by Nick Steinmetz
    (https://github.com/cortex-lab/sortingQuality) and converted to Python by Daniel Denman.

    Documentation / resources
    --------------------------
    For future documentation on isi violations, see:
    https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_quality_metrics.html#ISI-violations

    This documentation by Allen Brain provided information on what thresholds may be considered 
    acceptable for ISI violations.
    """

    assert isi_threshold_ms > 0, "The ISI threshold must be greater than 0."

    if (isi_threshold_ms > 1.5):
        warnings.warn("The ISI threshold is set to a value greater than the refractory period of most neurons.")

    isi_violations = {'isi_violations_ratio': np.nan, 'isi_violations_rate': np.nan,
                      'isi_violations_count': np.nan, 'isi_violations_of_total_spikes': np.nan}

    isi_threshold_s = isi_threshold_ms / 1000
    min_isi_s = min_isi_ms / 1000
    recording_length_s = recording_length_t / 1000

    isi_violations_count = {}
    isi_violations_ratio = {}

    # convert spike train to seconds
    if len(spike_train_ms) > 0:
        spike_train_s = spike_train_ms / 1000
    else:
        # if no spikes in the spike train, return nan
        return isi_violations

    isis = np.diff(spike_train_s)
    num_spikes = len(spike_train_s)
    num_violations = np.sum(isis < isi_threshold_s)

    violation_time = 2 * num_spikes * (isi_threshold_s - min_isi_s)

    if num_spikes > 0:
        total_rate = num_spikes / recording_length_s
        violation_rate = num_violations / violation_time
        isi_violations_ratio = violation_rate / total_rate
        isi_violations_proportion = (num_violations / recording_length_s)*100
        isi_violations_count = num_violations
        isi_violations_of_total = num_violations / num_spikes

    isi_violations['isi_violations_ratio'] = isi_violations_ratio
    isi_violations['isi_violations_proportion'] = isi_violations_proportion
    isi_violations['isi_violations_count'] = isi_violations_count
    isi_violations['isi_violations_of_total_spikes'] = isi_violations_of_total

    return isi_violations
