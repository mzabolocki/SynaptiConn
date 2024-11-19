""" Base model object, which is used to quantify monosynaptic connections between neurons. """

import warnings
import logging

import numpy as np
import pandas as pd

from typing import Any

from synapticonn.core.spike_times import SpikeManager
from synapticonn.plots import plot_acg, plot_ccg, plot_synaptic_strength
from synapticonn.monosynaptic_connections.synaptic_strength import calculate_synaptic_strength
from synapticonn.monosynaptic_connections.connection_type import get_putative_connection_type
from synapticonn.postprocessing.crosscorrelograms import compute_crosscorrelogram
from synapticonn.features import compute_peak_latency, compute_ccg_bootstrap, compute_ccg_cv, compute_peak_amp
from synapticonn.utils.errors import SpikeTimesError, ConnectionTypeError, DataError
from synapticonn.core.core_tools import extract_spike_unit_labels


###############################################################################
###############################################################################


class SynaptiConn(SpikeManager):
    """ Base class for quantifying monosynaptic connections between neurons.

    Parameters
    ----------
    spike_trains : dict
        Dictionary containing spike times for each unit (in milliseconds).
        Indexed by unit ID.
    bin_size_ms : float
        Bin size of the cross-correlogram (in milliseconds).
    max_lag_ms : float
        Maximum lag to compute the cross-correlogram (in milliseconds).
    recording_length_ms : float
        Length of the recording (in seconds).
    srate : float
        Sampling rate of the spike times (in Hz).

    Notes
    -----
    If spike trains are not in milliseconds, a conversion from seconds to milliseconds is attempted.

    Recording length is used to check if spike times exceed the recording duration. This is in 
    milliseconds to match the spike times.
    """


    ###########################################################################
    ###########################################################################


    def __init__(self, spike_times: dict = None,
                 bin_size_ms: float = 1,
                 max_lag_ms: float = 100,
                 recording_length_ms: float = None,
                 srate: float = None):
        """ Initialize the SynaptiConn object. """

        super().__init__(spike_times=spike_times,
                         srate=srate,
                         recording_length_ms=recording_length_ms)

        self.bin_size_ms = bin_size_ms
        self.max_lag_ms = max_lag_ms


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


    def reset_pair_synaptic_strength(self):
        """ Reset the synaptic strength data. """

        if hasattr(self, 'pair_synaptic_strength'):
            del self.pair_synaptic_strength
        else:
            raise DataError("No synaptic strength data found.")


    @extract_spike_unit_labels
    def synaptic_strength(self, spike_unit_labels: list, spike_pairs: list = None,
                          num_iterations: int = 1000, max_lag_ms: float = 25.0,
                          bin_size_ms: float = 0.5, jitter_range_ms: float = 10.0,
                          half_window_ms: float = 5, n_jobs: int = -1) -> dict:
        """ Compute the synaptic strength for the given spike pairs.

        Parameters
        ----------
        spike_unit_labels : list
            List of spike unit labels.
        spike_pairs : list
            List of spike pairs to compute synaptic strength.
            These are tuples of pre- and post-synaptic neuron IDs.
            Pre-synaptic neuron ID is the first element and post-synaptic neuron ID is the second element.
        num_iterations : int
            Number of iterations to compute the synaptic strength.
        max_lag_ms : float
            Maximum lag to compute the synaptic strength (in milliseconds).
        bin_size_ms : float
            Bin size of the synaptic strength (in milliseconds).
        jitter_range_ms : float
            Jitter range to compute the synaptic strength (in milliseconds).
        half_window_ms : float
            Half window size for the synaptic strength (in milliseconds).
        n_jobs: int
            Number of parallel jobs to run. Default is -1 (all cores).
            Use this to speed up computation.

        Returns
        -------
        synaptic_strength_pairs : dict
            Dictionary containing synaptic strength data for all pairs of spike trains.
            This contains the mean, standard deviation, and confidence intervals of the synaptic strength
            following jittering and bootstrapping.

        References
        ----------
        [1] STAR Protoc. 2024 Jun 21;5(2):103035. doi: 10.1016/j.xpro.2024.103035. Epub 2024 Apr 27.
        """

        valid_spike_pairs, _ = self._filter_spike_pairs(spike_pairs, spike_unit_labels)

        self.pair_synaptic_strength = {}
        for pre_synaptic_neuron_id, post_synaptic_neuron_id in valid_spike_pairs:

            # retrieve spike times for the pre- and post-synaptic neurons
            pre_synaptic_spktimes = self.get_spike_times_for_units([pre_synaptic_neuron_id]).get(pre_synaptic_neuron_id)
            post_synaptic_spktimes = self.get_spike_times_for_units([post_synaptic_neuron_id]).get(post_synaptic_neuron_id)

            # calculate synaptic strength
            synaptic_strength_data = calculate_synaptic_strength(pre_synaptic_spktimes,
                                                                 post_synaptic_spktimes,
                                                                 jitter_range_ms=jitter_range_ms,
                                                                 num_iterations=num_iterations,
                                                                 max_lag_ms=max_lag_ms,
                                                                 bin_size_ms=bin_size_ms,
                                                                 half_window_ms=half_window_ms,
                                                                 n_jobs=n_jobs)

            self.pair_synaptic_strength[(pre_synaptic_neuron_id, post_synaptic_neuron_id)] = synaptic_strength_data

        return self.pair_synaptic_strength


    def plot_synaptic_strength(self, spike_pair: tuple = None, **kwargs):
        """ Plot the synaptic strength for the given spike pair.

        Note, this method requires the synaptic strength data to be computed first.
        It only plots the synaptic strength for a single pair of spike trains.
        """

        assert spike_pair is not None, "Please provide a valid spike pair to plot."
        assert isinstance(spike_pair, tuple), "Spike pair must be a tuple."

        if not hasattr(self, 'pair_synaptic_strength'):
            raise DataError("No synaptic strength data found. Please run the synaptic_strength method first.")

        plot_synaptic_strength(self.pair_synaptic_strength, spike_pair, **kwargs)


    def monosynaptic_connection_types(self, threshold: float = None) -> dict:
        """ Categorize monosynaptic connection types based on synaptic strength data output.

        Parameters
        ----------
        threshold : float
            Threshold value for categorizing connection types. Default is None.

        Returns
        -------
        connection_types : dict
            Dictionary containing connection types for all pairs of spike trains.

        Notes
        -----
        Based on [1], for excitatory connections, a threshold of  5 is recommended.
        For inhibitory connections, a threshold of  -5 is recommended. Thresholds
        can be adjusted based on the synaptic strength data.

        References
        ----------
        [1] STAR Protoc. 2024 Jun 21;5(2):103035. doi: 10.1016/j.xpro.2024.103035. Epub 2024 Apr 27.
        """

        if hasattr(self, 'pair_synaptic_strength'):
            connection_types = {}
            for pair, synaptic_strength_data in self.pair_synaptic_strength.items():
                connection_types[pair] = get_putative_connection_type(synaptic_strength_data['synaptic_strength'], threshold=threshold)
            return connection_types
        else:
            raise ConnectionTypeError("No synaptic strength data found. Please run the synaptic_strength method first.")


    def monosynaptic_connection_features(self, n_boothstraps: int = 1000):
        """ Extract connection features from synaptic strength data. """

        if hasattr(self, 'pair_synaptic_strength'):
            connection_features = {}
            for pair, synaptic_strength_data in self.pair_synaptic_strength.items():
                peak_time = compute_peak_latency(synaptic_strength_data['original_ccg_counts'], self.bin_size_ms)
                peak_amp = compute_peak_amp(synaptic_strength_data['original_ccg_counts'])
                std_bootstrap = compute_ccg_bootstrap(synaptic_strength_data['original_ccg_counts'], n_bootstraps=n_boothstraps)
                cv_ccg = compute_ccg_cv(synaptic_strength_data['original_ccg_counts'])

                connection_features[pair] = {'synaptic_strength': synaptic_strength_data['synaptic_strength']}
                connection_features[pair].update(peak_time)
                connection_features[pair].update(peak_amp)
                connection_features[pair].update(std_bootstrap)
                connection_features[pair].update(cv_ccg)

            return connection_features
        else:
            raise ConnectionTypeError("No synaptic strength data found. Please run the synaptic_strength method first.")


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

        spike_units_to_collect = self._get_valid_spike_unit_labels(spike_units, spike_unit_labels)
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

        valid_spike_units = self._get_valid_spike_unit_labels(spike_pairs, spike_unit_labels)
        valid_spike_pairs, _ = self._filter_spike_pairs(spike_pairs, spike_unit_labels)

        # retrieve spike times and compute cross-correlogram data
        spike_times = self.get_spike_times_for_units(valid_spike_units)
        crosscorrelogram_data = compute_crosscorrelogram(
            spike_times, valid_spike_pairs, bin_size_ms=self.bin_size_ms, max_lag_ms=self.max_lag_ms)

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


    def _get_valid_spike_unit_labels(self, spike_units: list = None,
                                     spike_unit_labels: list = None):
        """ Validate and filter spike unit labels.

        Parameters
        ----------
        spike_units : list
            List of spike units to select for.
        spike_unit_labels : list
            List of spike unit labels.

        Returns
        -------
        spike_unit_labels : list
            List of valid spike units to plot.
        """

        if spike_units is None:
            raise SpikeTimesError('Please provide spike units to plot.')

        if not isinstance(spike_units, np.ndarray):
            spike_units = np.array(spike_units)

        spike_unit_labels = spike_units[np.isin(spike_units, spike_unit_labels)]
        if len(spike_unit_labels) == 0:
            raise SpikeTimesError('No valid spike units to plot.')

        return spike_unit_labels


    def _filter_spike_pairs(self, spike_pairs: list = None, spike_unit_labels: list = None):
        """ Filter spike pairs for valid spike units.

        Parameters
        ----------
        spike_pairs : list
            List of spike pairs.
        spike_unit_labels : list
            List of spike unit labels.

        Returns
        -------
        valid_spike_pairs : list
            List of valid spike pairs.
        invalid_spike_pairs : list
            List of invalid spike pairs.
        """

        valid_spike_units = self._get_valid_spike_unit_labels(spike_pairs, spike_unit_labels)

        invalid_spike_pairs = [pair for pair in spike_pairs if pair[0] not in valid_spike_units or pair[1] not in valid_spike_units]
        valid_spike_pairs = [pair for pair in spike_pairs if pair[0] in valid_spike_units and pair[1] in valid_spike_units]

        if invalid_spike_pairs:
            warnings.warn(
                f"Invalid spike pairs found: {invalid_spike_pairs}. These pairs will be ignored.",
                UserWarning
            )
        if not valid_spike_pairs:
            raise SpikeTimesError("No valid spike pairs found for the given spike unit labels.")

        return valid_spike_pairs, invalid_spike_pairs
