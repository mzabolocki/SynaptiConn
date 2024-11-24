""" isi_violations.py.

Modules for computing interspike interval (ISI) violations in a spike train.
"""


import warnings
import numpy as np


####################################################
####################################################


def compute_isi_violations(spike_train,
                           recording_length_t,
                           isi_threshold_ms=1.5,
                           min_isi_ms=0,
                           time_unit='ms'):
    """Compute the number of interspike interval (ISI) violations in a spike train.

    Parameters
    ----------
    spike_train : numpy.ndarray (1D)
        Spike train.
    recording_length_t : float
        Length of the recording.
    time_unit : str
        Time unit of the recording. Options are 'ms' or 's'.
    isi_threshold_ms : float
        Minimum interspike interval, in ms.
        This is the minimum time that must elapse between two spikes.
    min_isi_ms : float
        Minimum possible interspike interval in ms.
        This is the artifical refractory period enforced by the
        recording system or the spike sorting algorithm.

    Notes
    -----
    All times are in milliseconds (ms) by default.
    If not, the recording length and spike times will be converted to milliseconds.

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

    isi_violations = {'isi_violations_ratio': np.nan,
                      'isi_violations_count': np.nan,
                      'isi_violations_of_total_spikes': np.nan}

    # if the time unit is in seconds, warn the user
    # that the ISI violations will be skipped and
    # that the time unit should be converted to milliseconds
    if time_unit == 's':
        warnings.warn("The time unit is set to seconds. Spike times "
                      "will be converted to milliseconds to calculate ISI violations.")
        spike_times_ms = spike_train * 1000
        recording_length_ms = recording_length_t * 1000
    elif time_unit == 'ms':
        spike_times_ms = spike_train
        recording_length_ms = recording_length_t
    else:
        raise ValueError("Time unit must be 'ms' or 's'.")

    # warn the user if the ISI threshold is
    # greater than the refractory period
    if (isi_threshold_ms > 1.5):
        warnings.warn("The ISI threshold is set to a value greater than the "
                      "refractory period of most neurons.")

    isi_violations_count = {}
    isi_violations_ratio = {}

    isis = np.diff(spike_times_ms)
    num_spikes = len(spike_times_ms)
    num_violations = np.sum(isis < isi_threshold_ms)

    violation_time = 2 * num_spikes * (isi_threshold_ms - min_isi_ms)

    if num_spikes > 0:
        total_rate = num_spikes / (recording_length_ms / 1000)
        violation_rate = num_violations / violation_time
        isi_violations_ratio = violation_rate / total_rate
        isi_violations_count = num_violations
        isi_violations_of_total = num_violations / num_spikes

    isi_violations['isi_violations_ratio'] = isi_violations_ratio
    isi_violations['isi_violations_count'] = isi_violations_count
    isi_violations['isi_violations_of_total_spikes'] = isi_violations_of_total

    return isi_violations
