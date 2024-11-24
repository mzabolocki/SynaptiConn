""" unit_fr.py

Modules for computing unit firing rates.
"""


####################################################
####################################################


def compute_firing_rates(spike_train, recording_length_t, time_unit):
    """ Compute the firing rates of a spike train.

    Parameters
    ----------
    spike_train : numpy.ndarray (1D)
        Spike train of a single unit.
    recording_length_t : float
        Length of the recording.
    time_unit : str
        Time unit of the recording.
        Options are 'ms' or 's'.

    Returns
    -------
    firing_rates : dict
        Dictionary containing the firing rates
        and the number of spikes.

    Notes
    -----
    Firing rates are calculated as the number of spikes divided by the
    duration of the recording. The firing rate is given in Hz.
    """

    total_spikes = len(spike_train)

    if time_unit == "s":
        unit_fr = total_spikes / recording_length_t
    elif time_unit == "ms":
        unit_fr = total_spikes / (recording_length_t / 1000)
    else:
        raise ValueError("Time unit must be 'ms' or 's'.")

    return {"n_spikes": total_spikes, "firing_rate_hz": unit_fr}