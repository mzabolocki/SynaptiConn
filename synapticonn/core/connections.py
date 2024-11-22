""" Base model object, which is used to quantify monosynaptic connections between neurons. """

import warnings
import logging

import numpy as np
import pandas as pd

from typing import Any, List, Tuple

from synapticonn.core.spike_times import SpikeManager
from synapticonn.plots import plot_acg, plot_ccg, plot_ccg_synaptic_strength
from synapticonn.monosynaptic_connections.ccg_synaptic_strength import calculate_synaptic_strength
from synapticonn.monosynaptic_connections.ccg_connection_type import get_putative_connection_type
from synapticonn.postprocessing.crosscorrelograms import compute_crosscorrelogram
from synapticonn.features import compute_peak_latency, compute_ccg_bootstrap, compute_ccg_cv, compute_peak_amp
from synapticonn.utils.errors import SpikeTimesError, DataError
from synapticonn.core.core_tools import extract_spike_unit_labels
from synapticonn.utils.report import gen_model_results_str


###############################################################################
###############################################################################


class SynaptiConn(SpikeManager):
    """ Base class for quantifying monosynaptic connections between neurons.

    Parameters
    ----------
    spike_trains : dict
        Dictionary containing spike times for each unit indexed by unit ID.
        Spike times must be a float array.
    time_unit : str
        Time unit options in ms (milliseconds) or s (seconds).
        These are used to set the time unit for the spike times, recording length, 
        bin size, and maximum lag for all processing.
    bin_size_t : float
        Bin size of the cross-correlogram (in milliseconds).
    max_lag_t : float
        Maximum lag to compute the cross-correlogram (in milliseconds).
    method : str
        Type of synaptic strength to compute. Default is 'cross-correlation'.
        This performs the following:
            1. a peak detection on the cross-correlogram to estimate the synaptic strength
            2. a statistical analysis to estimate the confidence intervals
            3. a jittering analysis to estimate the jittered synaptic strength.
        In future versions, this will be expanded to include other types of correlation methods.
    recording_length_t : float
        Length of the recording.
    srate : float
        Sampling rate of the spike times (in Hz).
    spike_id_type : type
        Data type of the spike IDs.

    Notes
    -----
    If spike trains are not in milliseconds, a conversion from seconds to milliseconds is attempted.

    Recording length is used to check if spike times exceed the recording duration. This is in 
    milliseconds to match the spike times.
    """


    ###########################################################################
    ###########################################################################


    def __init__(self, spike_times: dict = None,
                 time_unit: str = 'ms',
                 bin_size_t: float = 1,
                 max_lag_t: float = 100,
                 method: str = 'cross-correlation',
                 recording_length_t: float = None,
                 srate: float = None,
                 spike_id_type: type = int):
        """ Initialize the SynaptiConn object. """

        super().__init__(spike_times=spike_times,
                         time_unit=time_unit,
                         srate=srate,
                         recording_length_t=recording_length_t,
                         spike_id_type=spike_id_type)

        self.bin_size_t = bin_size_t
        self.max_lag_t = max_lag_t
        self.method = method


    def report_correlogram_settings(self):
        """ Report the bin settings. """

        return f"Bin size: {self.bin_size_t} ms, Max lag: {self.max_lag_t} ms"


    def _get_default_settings(self):
        """ Return the settings of the object. """

        settings = {
            'bin_size_t': self.bin_size_t,
            'max_lag_t': self.max_lag_t,
            'method': self.method,
            'recording_length_t': self.recording_length_t,
            'srate': self.srate,
            'spike_id_type': self.spike_id_type
            }

        if self.method == 'cross-correlation':  # default settings for the cross-correlation method

            crosscorr_connection_settings = {
                'bin_size_t': 1,
                'max_lag_t': 100,
                'num_iterations': 1000,
                'jitter_range_ms': 10,
                'half_window_ms': 5,
                'n_jobs': -1
            }

            settings.update(crosscorr_connection_settings)

            return settings

        else:
            raise NotImplementedError("Only the 'cross-correlation' method is currently implemented.")


    def set_bin_settings(self, bin_size_t: float = 1, max_lag_t: float = 100):
        """ Set the settings of the object.

        Useful for changing the bin size and maximum lag after initialization.

        Parameters
        ----------
        bin_size_t : float
            Bin size of the cross-correlogram (in milliseconds) or auto-correlograms.
        max_lag_t : float
            Maximum lag to compute the cross-correlogram (in milliseconds).
        """

        self.bin_size_t = bin_size_t
        self.max_lag_t = max_lag_t
        self._run_initial_spike_time_val_checks()


    def reset_pair_synaptic_strength(self):
        """ Reset the synaptic strength data. """

        if hasattr(self, 'pair_synaptic_strength'):
            del self.pair_synaptic_strength
        else:
            raise DataError("No synaptic strength data found.")


    def fit(self, spike_pairs: List[Tuple] = None,
            synaptic_strength_threshold: float = 5,
            **kwargs) -> dict:
        """ Compute monosynaptic connections between neurons for a given set of spike times.

        Parameters
        ----------
        spike_pairs : List[Tuple]
            List of spike pairs to compute the synaptic strength.
            These are tuples of pre- and post-synaptic neuron IDs.
            Pre-synaptic neuron ID is the first element and post-synaptic neuron ID is the second element.
        synaptic_strength_threshold : float
            Threshold value for categorizing connection types. Default is 5.
            This is used to categorize the connection types based on the synaptic strength values.
        **kwargs : dict, optional
            Additional parameters for customizing the computation. Includes:
            - num_iterations : int
                Number of iterations for computing synaptic strength (default: 1000).
            - max_lag_t : float
                Maximum lag to compute the synaptic strength (in ms, default: 25.0).
            - bin_size_t : float
                Bin size for computing the synaptic strength (in ms, default: 0.5).
            - jitter_range_ms : float
                Jitter range for synaptic strength computation (in ms, default: 10.0).
            - half_window_ms : float
                Half window size for synaptic strength computation (in ms, default: 5).
            - n_jobs : int
                Number of parallel jobs to use (default: -1, all cores).

        Attributes set
        --------------
        pair_synaptic_strength : dict
            Dictionary containing the synaptic strength for each pair of neurons.
            This is stored in the object for future reference, and can be accessed using the 'pair_synaptic_strength' attribute.
            This is used to compute the connection types and features, and perform visualizations.

        Raises
        ------
        SpikeTimesError
            If spike pairs are not provided.
        DataError
            If no synaptic strength data is found.
        """

        # check if spike pairs are provided
        if spike_pairs is None:
            raise ValueError("Please provide a list of spike pairs to compute the connections.")

        # check if spike pairs are of a valid type
        if not isinstance(spike_pairs, List):
            raise SpikeTimesError("Spike pairs must be a list of tuples.")
        elif not all(isinstance(pair, Tuple) for pair in spike_pairs):
            raise SpikeTimesError("Spike pairs must be a list of tuples.")

        # compute and set the synaptic strength for the given spike pairs
        synaptic_strength_data = self.synaptic_strength(spike_pairs=spike_pairs, **kwargs)

        # isolate the mono-synaptic connections
        connection_types = self.monosynaptic_connection_types(synaptic_strength_threshold)

        # extract connection features
        # the number of bootstraps can be adjusted, but the default is 1000
        connection_features = self.monosynaptic_connection_features(kwargs.get('n_bootstraps', 1000))

        # merge the connection types and features
        for pair in connection_types:
            connection_types[pair].update(connection_features[pair])

        return connection_types


    def report(self, spike_pairs: List[Tuple] = None,
               synaptic_strength_threshold: float = 5,
               concise: bool = False,
               **kwargs):
        """ Compute the synaptic strength and connection types, and display a report.

        Parameters
        ----------
        spike_pairs : List[Tuple]
            List of spike pairs to compute the synaptic strength.
            These are tuples of pre- and post-synaptic neuron IDs.
            Pre-synaptic neuron ID is the first element and post-synaptic neuron ID is the second element.
        synaptic_strength_threshold : float
            Threshold value for categorizing connection types. Default is 5.
            This is used to categorize the connection types based on the synaptic strength values.
        concise : bool
            If True, print a concise summary of the results. This excludes blank lines.
        **kwargs : dict, optional
            Additional parameters for customizing the computation. Includes:
            - num_iterations : int
                Number of iterations for computing synaptic strength (default: 1000).
            - max_lag_t : float
                Maximum lag to compute the synaptic strength (in ms, default: 25.0).
            - bin_size_t : float
                Bin size for computing the synaptic strength (in ms, default: 0.5).
            - jitter_range_ms : float
                Jitter range for synaptic strength computation (in ms, default: 10.0).
            - half_window_ms : float
                Half window size for synaptic strength computation (in ms, default: 5).
            - n_jobs : int
                Number of parallel jobs to use (default: -1, all cores).

        Notes
        -----
        Data is computed and displayed in a report format.

        Attributes set
        --------------
        pair_synaptic_strength : dict
            Dictionary containing the synaptic strength for each pair of neurons.
            This is stored in the object for future reference, and can be accessed using the 'pair_synaptic_strength' attribute.
            This is used to compute the connection types and features, and perform visualizations.
        """

        # find default settings for reporting
        # and update with any additional parameters passed
        settings = {**self._get_default_settings(), **kwargs}
        settings['synaptic_strength_threshold'] = synaptic_strength_threshold

        # compute the synaptic strength and connection types
        connection_types = self.fit(spike_pairs, synaptic_strength_threshold, **kwargs)

        # print the results
        self.print_connection_results(connection_types, concise, settings)


    def print_connection_results(self, connection_types: dict = None, concise: bool = False, params: dict = None):
        """ Print the results of the synaptic strength and connection types.

        Parameters
        ----------
        connection_types : dict
            Dictionary containing connection types for all pairs of spike trains.
            This is computed using the 'fit' method.
        concise : bool
            If True, print a concise summary of the results.
            If False, print a detailed summary of the results.
        params : dict
            Additional parameters used for computing the model.
        """

        print(gen_model_results_str(connection_types, concise, params))


    @extract_spike_unit_labels
    def synaptic_strength(self, spike_unit_labels: list,
                          spike_pairs: List[Tuple] = None,
                          num_iterations: int = 1000,
                          max_lag_t: float = 25.0,
                          bin_size_t: float = 0.5,
                          jitter_range_ms: float = 10.0,
                          half_window_ms: float = 5,
                          n_jobs: int = -1) -> dict:
        """ Compute the synaptic strength for the given spike pairs.

        Parameters
        ----------
        spike_unit_labels : list
            List of spike unit labels.
        spike_pairs : List[Tuple]
            List of spike pairs to compute synaptic strength.
            These are tuples of pre- and post-synaptic neuron IDs.
            Pre-synaptic neuron ID is the first element and post-synaptic neuron ID is the second element.
        num_iterations : int
            Number of iterations to compute the synaptic strength.
        max_lag_t : float
            Maximum lag to compute the synaptic strength (in milliseconds).
        bin_size_t : float
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

        Attributes set
        --------------
        pair_synaptic_strength : dict
            Dictionary containing the synaptic strength for each pair of neurons.
            This is stored in the object for future reference, and can be accessed using the 'pair_synaptic_strength' attribute.
            This is used to compute the connection types and features, and perform visualizations.

        References
        ----------
        [1] STAR Protoc. 2024 Jun 21;5(2):103035. doi: 10.1016/j.xpro.2024.103035. Epub 2024 Apr 27.

        Notes
        -----
        This method computes the synaptic strength for all pairs of spike trains. Currently, 
        only the cross-correlogram method is implemented. In future versions, this will be expanded
        to include other types of correlation methods, such as cross-correlation, mutual information, etc.

        The 'cross-correlation' method computes the synaptic strength using the cross-correlogram. This method
        performs the following:
            1. a peak detection on the cross-correlogram to estimate the synaptic strength
            2. a statistical analysis to estimate the confidence intervals
            3. a jittering analysis to estimate the jittered synaptic strength.

        Analysis is based on [1]. For excitatory connections, a threshold of 5 is recommended.
        """

        # check if the method is implemented
        # note that only the 'cross-correlation' method is currently implemented
        # and for future versions, this will be expanded to include other types of correlation methods
        if self.method != 'cross-correlation':
            raise NotImplementedError("Only the 'cross-correlation' method is currently implemented. Please choose this method.")

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
                                                                 max_lag_t=max_lag_t,
                                                                 bin_size_t=bin_size_t,
                                                                 half_window_ms=half_window_ms,
                                                                 n_jobs=n_jobs)

            self.pair_synaptic_strength[(pre_synaptic_neuron_id, post_synaptic_neuron_id)] = synaptic_strength_data

        return self.pair_synaptic_strength


    def monosynaptic_connection_types(self, synaptic_strength_threshold: float = None) -> dict:
        """ Categorize monosynaptic connection types based on synaptic strength data output.

        Parameters
        ----------
        synaptic_strength_threshold : float
            Threshold value for categorizing connection types. Default is None.
            This is used to categorize the connection types based on the synaptic strength values.

        Returns
        -------
        connection_types : dict
            Dictionary containing connection types for all pairs of spike trains.

        Notes
        -----
        Based on [1], for excitatory connections, a threshold of  5 is recommended.
        For inhibitory connections, a threshold of  -5 is recommended. Thresholds
        can be adjusted based on the synaptic strength data.

        Currently, connection types are based on the synaptic strength values. This is 
        computed using the cross-correlogram method. In future versions, this will be expanded
        to include other types of correlation methods, such as cross-correlation, mutual information, etc.
        Please see [1] for more details.

        References
        ----------
        [1] STAR Protoc. 2024 Jun 21;5(2):103035. doi: 10.1016/j.xpro.2024.103035. Epub 2024 Apr 27.
        """

        if hasattr(self, 'pair_synaptic_strength'):
            connection_types = {}
            for pair, synaptic_strength_data in self.pair_synaptic_strength.items():
                connection_types[pair] = get_putative_connection_type(synaptic_strength_data['synaptic_strength'],
                                                                      threshold=synaptic_strength_threshold)
            return connection_types
        else:
            raise DataError("No synaptic strength data found. Please run the synaptic_strength method first.")


    def monosynaptic_connection_features(self, n_bootstraps: int = 1000) -> dict:
        """ Extract connection features from synaptic strength data.

        Parameters
        ----------
        n_bootstraps : int
            Number of bootstraps to compute the confidence intervals.

        Returns
        -------
        connection_features : dict
            Dictionary containing connection features for all pairs of spike trains.
        """

        if hasattr(self, 'pair_synaptic_strength'):
            connection_features = {}
            for pair, synaptic_strength_data in self.pair_synaptic_strength.items():
                peak_time = compute_peak_latency(synaptic_strength_data['original_crosscorr_counts'], self.bin_size_t)
                peak_amp = compute_peak_amp(synaptic_strength_data['original_crosscorr_counts'])
                std_bootstrap = compute_ccg_bootstrap(synaptic_strength_data['original_crosscorr_counts'], n_bootstraps=n_bootstraps)
                cv_crosscorr = compute_ccg_cv(synaptic_strength_data['original_crosscorr_counts'])

                connection_features[pair] = {'synaptic_strength': synaptic_strength_data['synaptic_strength']}
                connection_features[pair].update(peak_time)
                connection_features[pair].update(peak_amp)
                connection_features[pair].update(std_bootstrap)
                connection_features[pair].update(cv_crosscorr)

            return connection_features
        else:
            raise DataError("No synaptic strength data found. Please run the synaptic_strength method first.")


    def plot_synaptic_strength(self, spike_pair: tuple = None, **kwargs):
        """ Plot the synaptic strength for the given spike pair.

        Note, this method requires the synaptic strength data to be computed first.
        It only plots the synaptic strength for a single pair of spike trains.
        """

        assert spike_pair is not None, "Please provide a valid spike pair to plot."
        assert isinstance(spike_pair, tuple), "Spike pair must be a tuple."

        if not hasattr(self, 'pair_synaptic_strength'):
            raise DataError("No synaptic strength data found. Please run the synaptic_strength method first.")

        # check if the method is implemented
        # note that only the 'cross-correlation' method is currently implemented
        # and for future versions, this will be expanded to include other types of correlation methods
        if self.method == 'cross-correlation':
            plot_ccg_synaptic_strength(self.pair_synaptic_strength, spike_pair, **kwargs)
        else:
            raise NotImplementedError("Only the 'cross-correlation' method is currently implemented. Please choose this method.")


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
        plot_acg(spike_times, bin_size_t=self.bin_size_t, max_lag_t=self.max_lag_t, **kwargs)


    @extract_spike_unit_labels
    def return_crosscorrelogram_data(self, spike_unit_labels: list, spike_pairs: List[Tuple] = None) -> dict:
        """ Compute and return the cross-correlogram data for valid spike pairs.

        Parameters
        ----------
        spike_unit_labels : list
            List of spike unit labels (in strings).
        spike_pairs : List[Tuple]
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
            spike_times, valid_spike_pairs, bin_size_t=self.bin_size_t, max_lag_t=self.max_lag_t)

        return crosscorrelogram_data


    @extract_spike_unit_labels
    def plot_crosscorrelogram(self, spike_unit_labels: list, spike_pairs: List[Tuple] = None, **kwargs: Any):
        """ Plot the cross-correlogram for valid spike pairs.

        Parameters
        ----------
        spike_unit_labels : list
            List of spike unit labels (in strings).
        spike_pairs : List[Tuple]
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


    def _filter_spike_pairs(self, spike_pairs: List[Tuple] = None, spike_unit_labels: list = None):
        """ Filter spike pairs for valid spike units.

        Parameters
        ----------
        spike_pairs : List[Tuple]
            List of spike pairs.
        spike_unit_labels : list
            List of spike unit labels.

        Returns
        -------
        valid_spike_pairs : List[Tuple]
            List of valid spike pairs.
        invalid_spike_pairs : List[Tuple]
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
