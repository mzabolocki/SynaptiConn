""" Base model object, which is used to quantify monosynaptic connections between neurons. """


import numpy as np

from functools import wraps
from typing import List, Optional, Dict, Any

from synapticonn.utils.attribute_checks import requires_sampling_rate, requires_recording_length
from synapticonn.plots.acg import plot_acg
from synapticonn.plots.ccg import plot_ccg
from synapticonn.postprocessing.crosscorrelograms import compute_crosscorrelogram
from synapticonn.utils.errors import SpikeTimesError


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
    recording_length : float
        Length of the recording (in seconds).
    srate : float
        Sampling rate of the spike times (in Hz).
    """


    # ----- CLASS VARIABLES
    # flag to check spike time conversion to milliseconds
    converted_to_ms = False


    ###########################################################################
    ###########################################################################


    def __init__(self, spike_times: dict = None,
                 bin_size_ms: float = 1,
                 max_lag_ms: float = 100,
                 recording_length: float = None,
                 srate: float = None):
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

        self._check_spike_times_type()
        self._check_spike_time_conversion()
        self._check_negative_spike_times()
        self._check_spike_times_values()


    def report_spike_units(self):
        """ Report the spike units. """

        labels = self.get_spike_unit_labels()
        n_spks = [len(self.spike_times[label]) for label in labels]
        spk_unit_summary = {'unit_id': labels, 'n_spikes': n_spks}

        return spk_unit_summary


    def report_correlogram_settings(self):
        """ Report the bin settings. """

        return f"Bin size: {self.bin_size_ms} ms, Max lag: {self.max_lag_ms} ms"


    def set_bin_settings(self, bin_size_ms: float = 1, max_lag_ms: float = 100):
        """ Set the settings of the object.

        Useful for changing the bin size and maximum lag after initialization.

        Parameters
        ----------
        bin_size_ms : float
            Bin size of the cross-correlogram (in milliseconds) or auto-correlograms.
        max_lag_ms : float
            Maximum lag to compute the cross-correlogram (in milliseconds).
        """

        self.bin_size_ms = bin_size_ms
        self.max_lag_ms = max_lag_ms
        self._run_initial_spike_time_checks()


    def get_spike_unit_labels(self):
        """ Retrieve the spike unit labels. """

        return list(self.spike_times.keys())


    def get_spike_times_for_units(self, spike_units_to_collect: list = None) -> dict:
        """ Retrieve spike times for the selected units.

        Parameters
        ----------
        spike_units_to_collect : list
            List of spike units to collect.

        Returns
        -------
        spike_times : dict
            Dictionary containing spike times for the selected units.
        """
        return {key: self.spike_times[key] for key in spike_units_to_collect}


    def _reset_parameters(self):
        """ Reset the parameters of the object. """

        self.spike_times = None
        self.bin_size_ms = None
        self.max_lag_ms = None
        self.recording_length = None
        self.srate = None


    @staticmethod
    def extract_spike_unit_labels(func):
        """ Decorator to inject spike unit labels from spike_times dictionary if not already provided. """

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # check if spike_unit_labels is provided in args or kwargs
            if 'spike_unit_labels' not in kwargs and len(args) < func.__code__.co_argcount - 1:
                # if not present in kwargs and missing in positional args, add to kwargs
                kwargs['spike_unit_labels'] = list(self.spike_times.keys())
            return func(self, *args, **kwargs)

        return wrapper


    @extract_spike_unit_labels
    def plot_autocorrelogram(self, spike_unit_labels: list,
                             spike_units: list = None, **kwargs):
        """ Plot the autocorrelogram.

        Parameters
        ----------
        spike_unit_labels : list
            List of spike unit labels.
        spike_units : list
            List of spike units to plot.
        **kwargs
            Additional keyword arguments passed to `plot_acg`.

        Notes
        -----
        Autocorrelograms are computed for each spike unit and plotted.
        The bin size and maximum lag are set by the object parameters.
        """

        spike_units_to_collect = self._get_valid_spike_units(spike_units, spike_unit_labels)
        print(f'Plotting autocorrelogram for spike units: {spike_units_to_collect}')

        spike_times = self.get_spike_times_for_units(spike_units_to_collect)
        plot_acg(spike_times, bin_size_ms=self.bin_size_ms, max_lag_ms=self.max_lag_ms, **kwargs)


    @extract_spike_unit_labels
    def return_crosscorrelogram_data(self, spike_unit_labels: list, spike_pairs: list = None) -> dict:
        """ Compute and return the cross-correlogram data for valid spike pairs.

        Parameters
        ----------
        spike_unit_labels : list
            List of spike unit labels (in strings).
        spike_pairs : list
            List of spike pairs to compute the cross-correlogram data.

        Returns
        -------
        crosscorrelogram_data : dict
            Dictionary containing cross-correlograms and bins for all pairs of spike trains.
        """

        # validate spike pairs
        valid_spike_units = self._get_valid_spike_units(np.unique(spike_pairs), spike_unit_labels)
        filtered_spike_pairs = [pair for pair in spike_pairs if pair[0] in valid_spike_units and pair[1] in valid_spike_units]
        filtered_spike_pairs = list(set(filtered_spike_pairs))  # remove duplicates

        if not filtered_spike_pairs:
            raise ValueError("No valid spike pairs found for the given spike unit labels.")

        # retrieve spike times and compute cross-correlogram data
        spike_times = self.get_spike_times_for_units(valid_spike_units)
        crosscorrelogram_data = compute_crosscorrelogram(
            spike_times, filtered_spike_pairs, bin_size_ms=self.bin_size_ms, max_lag_ms=self.max_lag_ms)

        return crosscorrelogram_data


    @extract_spike_unit_labels
    def plot_crosscorrelogram(self, spike_unit_labels: list, spike_pairs: list = None, **kwargs: Any):
        """ Plot the cross-correlogram for valid spike pairs.

        Parameters
        ----------
        spike_unit_labels : list
            List of spike unit labels (in strings).
        spike_pairs : list
            List of spike pairs to plot.
        **kwargs : Any
            Additional keyword arguments passed to `plot_ccg`.
        """

        crosscorrelogram_data = self.return_crosscorrelogram_data(spike_unit_labels, spike_pairs)
        plot_ccg(crosscorrelogram_data, **kwargs)


    @requires_sampling_rate
    @requires_recording_length
    def _check_spike_time_conversion(self):
        """ Check that spike time values are in millisecond format. """

        if SynaptiConn.converted_to_ms:
            try:
                # check for spike times type incase it was changed
                # with object re-initialization
                self._check_spike_times_values()
            except SpikeTimesError:
                # if spike times are not in milliseconds, then convert
                # if this does not work, _run_initial_spike_time_checks
                # will raise an error
                pass
            else:
                return

        converted_keys = []
        for key, spks in self.spike_times.items():
            if len(spks) == 0:
                raise SpikeTimesError(f"Spike times for unit {key} are empty.")

            max_spk_time = np.max(spks)
            recording_length_ms = self.recording_length * 1000

            # check if spike times need to be converted to milliseconds
            if max_spk_time > recording_length_ms:
                self.spike_times[key] = (spks / self.srate) * 1000
                converted_keys.append(key)
            elif max_spk_time > self.recording_length:
                raise SpikeTimesError(f"Spike times for unit {key} exceed the recording length after conversion.")

        if converted_keys:
            converted_keys_str = ', '.join(map(str, converted_keys))
            print(f"Warning: Spike times for unit(s) {converted_keys_str} were converted to milliseconds.")

        SynaptiConn.converted_to_ms = True


    def _check_negative_spike_times(self):
        """ Check for negative spike times. """

        for key, spks in self.spike_times.items():
            if not np.all(spks >= 0):
                raise SpikeTimesError(f'Spike times for unit {key} must be non-negative.')


    def _check_spike_times_type(self):
        """ Ensure spike times is a dictionary. """

        if not isinstance(self.spike_times, dict):
            raise SpikeTimesError('Spike times must be a dictionary with unit-ids as keys.')


    def _check_spike_times_values(self):
        """ Check the values of the spike times dictionary are in floats or arr format. """

        for key, value in self.spike_times.items():
            if not isinstance(value, np.ndarray):
                raise SpikeTimesError(f'Spike times for unit {key} must be a 1D numpy array. Got {type(value)} instead.')
            if not np.issubdtype(value.dtype, np.floating):
                raise SpikeTimesError(f'Spike times for unit {key} must be a 1D array of floats. Got {type(value)} instead.')


    def _get_valid_spike_units(self, spike_units_to_plot: list = None,
                               spike_unit_labels: list = None):
        """ Validate and filter spike units to plot based on available labels.

        Parameters
        ----------
        spike_units_to_plot : list
            List of spike units to plot.
        spike_unit_labels : list
            List of spike unit labels.

        Returns
        -------
        spike_units_to_plot : list
            List of valid spike units to plot.
        """

        if spike_units_to_plot is None:
            raise SpikeTimesError('Please provide spike units to plot.')

        if not isinstance(spike_units_to_plot, np.ndarray):
            spike_units_to_plot = np.array(spike_units_to_plot)

        spike_units_to_plot = spike_units_to_plot[np.isin(spike_units_to_plot, spike_unit_labels)]
        if len(spike_units_to_plot) == 0:
            raise SpikeTimesError('No valid spike units to plot.')

        return spike_units_to_plot

