""" Base model object, which is used to quantify monosynaptic connections between neurons. """


import numpy as np
import pandas as pd

from synapticonn.utils.attribute_checks import requires_sampling_rate, requires_recording_length


###############################################################################
###############################################################################


class SynaptiConn():
    """ Base class for quantifying monosynaptic connections between neurons.

    Parameters
    ----------
    spike_trains : dict
        Dictionary containing spike times for each unit.
        Indexed by unit ID.
    bin_size_ms : float
        Bin size of the cross-correlogram (in milliseconds).
    max_lag_ms : float
        Maximum lag to compute the cross-correlogram (in milliseconds).
    srate : float
        Sampling rate of the spike times (in Hz).
    """

    # ----- CLASS VARIABLES
    # flag to check spike time conversion to milliseconds
    conversion = False


    ###########################################################################
    ###########################################################################


    def __init__(self, spike_times, bin_size_ms=1, max_lag_ms=100, recording_length=None, srate=None):
        """ Initialize the SynaptiConn object. """

        self.spike_times = spike_times
        self.bin_size_ms = bin_size_ms
        self.max_lag_ms = max_lag_ms
        self.recording_length = recording_length
        self.srate = srate

        # internal checks
        self._run_initial_spike_time_checks()

    def _run_initial_spike_time_checks(self):
        """ Run all the spike-time-related checks at initialization. """
        self._check_spike_time_conversion()
        self._check_negative_spike_times()
        self._check_spike_times_type()

    @requires_sampling_rate
    @requires_recording_length
    def _check_spike_time_conversion(self):
        """ Check that spike time values are in millisecond format. """

        if SynaptiConn.conversion:
            return

        converted_keys = []
        for key, spks in self.spike_times.items():
            if len(spks) == 0:
                raise ValueError(f"Spike times for unit {key} are empty.")

            max_spk_time = np.max(spks)
            recording_length_ms = self.recording_length * 1000

            # check if spike times need to be converted to milliseconds
            if max_spk_time > recording_length_ms:
                self.spike_times[key] = (spks / self.srate) * 1000
                converted_keys.append(key)
            elif max_spk_time > self.recording_length:
                raise ValueError(f"Spike times for unit {key} exceed the recording length after conversion.")

        if converted_keys:
            converted_keys_str = ', '.join(map(str, converted_keys))
            print(f"Warning: Spike times for unit(s) {converted_keys_str} were converted to milliseconds.")

        SynaptiConn.conversion = True

    def _check_negative_spike_times(self):
        """ Check for negative spike times. """

        for key, spks in self.spike_times.items():
            if not np.all(spks >= 0):
                raise ValueError(f'Spike times for unit {key} must be non-negative.')

    def _check_spike_times_type(self):
        """ Check the spike times type for correctness. """

        if not isinstance(self.spike_times, dict):
            raise ValueError('Spike times must be a dictionary with unit-ids as keys and numpy arrays as values.')

        # check that all values in the dictionary are numpy arrays of floats
        for key, value in self.spike_times.items():
            if not isinstance(value, np.ndarray):
                raise ValueError(f'Spike times for unit {key} must be a numpy array.')
            if not np.issubdtype(value.dtype, np.floating):
                raise ValueError(f'Spike times for unit {key} must be an array of floats.')



# self.cross_correlograms_data = self.compute_crosscorrelogram()
# repeat this for ACGs also ...
# check the types for the inputs here
# report summary on load?
# option to set spike train set
# check the quality of the acgs etc.
# option to drop these if below the threshold cut-offs and keep a log of this???