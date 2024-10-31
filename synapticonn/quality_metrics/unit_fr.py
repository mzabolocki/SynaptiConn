""" unit_fr.py

Modules for computing unit firing rates.
"""


####################################################
####################################################


def compute_firing_rates(spike_train_ms, recording_length_ms):
    """ Compute the firing rates of a spike train.

    Parameters
    ----------
    spike_train_ms : numpy.ndarray
        Spike train in milliseconds.
    recording_length_ms : float
        Length of the recording in milliseconds.

    Returns
    -------
    firing_rates : dict
        Dictionary containing the firing rates.

    Notes
    -----
    Firing rates are calculated as the number of spikes divided by the
    duration of the recording. The firing rate is given in Hz.
    """

    total_spikes = len(spike_train_ms)

    unit_fr = total_spikes / recording_length_ms * 1000  # in Hz

    return {"recording_length_sec": recording_length_ms/1000, "n_spikes": total_spikes, "firing_rate_hz": unit_fr}