""" Base model object, which is used to quantify monosynaptic connections between neurons. """

import warnings
import logging
import pathlib

import numpy as np
import pandas as pd

from functools import wraps
from typing import List, Optional, Dict, Any

from synapticonn.utils.attribute_checks import requires_sampling_rate, requires_recording_length
from synapticonn.plots import plot_acg, plot_ccg, plot_synaptic_strength
from synapticonn.monosynaptic_connections.synaptic_strength import calculate_synaptic_strength
from synapticonn.monosynaptic_connections.connection_type import get_putative_connection_type
from synapticonn.postprocessing.crosscorrelograms import compute_crosscorrelogram
from synapticonn.quality_metrics import compute_isi_violations, compute_presence_ratio, compute_firing_rates
from synapticonn.utils.errors import SpikeTimesError, ConnectionTypeError, DataError, RecordingLengthError


###############################################################################
###############################################################################


class SynaptiConn():
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


    # ----- CLASS VARIABLES
    # flag to check spike time conversion to milliseconds
    converted_to_ms = False
    # spike unit filtering
    spike_unit_filtering = False

    # quality metric keys
    quality_metric_keys = ['isi_violations_ratio', 'isi_violations_rate', 'isi_violations_count',
                            'isi_violations_of_total_spikes', 'presence_ratio', 'firing_rate_hz',
                            'recording_length_sec', 'n_spikes']


    ###########################################################################
    ###########################################################################


    def __init__(self, spike_times: dict = None,
                 bin_size_ms: float = 1,
                 max_lag_ms: float = 100,
                 recording_length_ms: float = None,
                 srate: float = None):
        """ Initialize the SynaptiConn object. """

        self.spike_times = spike_times
        self.bin_size_ms = bin_size_ms
        self.max_lag_ms = max_lag_ms
        self.recording_length_ms = recording_length_ms
        self.srate = srate

        # internal checks
        self._run_initial_spike_time_checks()


    def _run_initial_spike_time_checks(self):
        """ Run all the spike-time-related checks at initialization. """

        self._check_spike_times_type()
        self._check_spike_time_conversion()
        self._check_negative_spike_times()
        self._check_spike_times_values()
        self._check_recording_length_ms()


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


    def set_spike_times(self, spike_times: dict):
        """ Set the spike times for the object.

        Parameters
        ----------
        spike_times : dict
            Dictionary containing spike times for each unit.
            Indexed by unit ID.
        """

        self.spike_times = spike_times
        self._run_initial_spike_time_checks()


    def reset_pair_synaptic_strength(self):
        """ Reset the synaptic strength data. """

        if hasattr(self, 'pair_synaptic_strength'):
            del self.pair_synaptic_strength
        else:
            raise DataError("No synaptic strength data found.")

    # TO DO :: move to utils?
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


    def spike_unit_quality(self,
                           isi_threshold_ms=1.5,
                           min_isi_ms=0,
                           presence_ratio_bin_duration_ms=60000,
                           presence_ratio_mean_fr_ratio_thresh=0.0) -> pd.DataFrame:
        """ Compute spike isolation quality metrics.

        Parameters
        ----------
        isi_threshold_ms : float
            Threshold for the interspike interval (ISI) violations.
        min_isi_ms : float
            Minimum ISI value (in milliseconds).
        presence_ratio_bin_duration_ms : float
            Duration of each bin in milliseconds for the presence ratio.
        presence_ratio_mean_fr_ratio_thresh : float
            Minimum mean firing rate ratio threshold for the presence ratio.
            This is the minimum mean firing rate that must be present in a bin
            for the unit to be considered "present" in that bin.
            By default, this is set to 0.0. This means that the unit must have
            at least one spike in each bin to be considered "present."

        Returns
        -------
        quality_metrics : pd.DataFrame
            DataFrame containing the quality metrics for each spike unit.

        Notes
        -----
        Quality metrics include:
        - isi_violations_ratio: Fraction of ISIs that violate the threshold.
        - isi_violations_rate: Rate of ISIs that violate the threshold.
        - isi_violations_count: Number of ISIs that violate the threshold.
        - isi_violations_of_total_spikes: Fraction of ISIs that violate the threshold out of total spikes.
        - presence_ratio: Fraction of time during a session in which a unit is spiking.
        - mean_firing_rate: Mean firing rate of the unit.
        - recording_length_sec: Length of the recording in seconds.
        - n_spikes: Number of spikes for the unit.

        For further information on the quality metric calculations,
        see the respective functions in the quality_metrics module.
        """

        quality_metrics = {}
        for key, spks in self.spike_times.items():

            # isi violations
            isi_violations = compute_isi_violations(spks, self.recording_length_ms,
                                                    isi_threshold_ms, min_isi_ms)

            # presence ratio
            presence_ratio = compute_presence_ratio(spks, self.recording_length_ms,
                                                    bin_duration_ms=presence_ratio_bin_duration_ms,
                                                    mean_fr_ratio_thresh=presence_ratio_mean_fr_ratio_thresh,
                                                    srate=self.srate)

            # unit firing rates
            firing_rates = compute_firing_rates(spks, self.recording_length_ms)

            quality_metrics[key] = isi_violations
            quality_metrics[key].update(presence_ratio)
            quality_metrics[key].update(firing_rates)

        return pd.DataFrame(quality_metrics).T


    def filter_spike_units(self, quality_metrics: pd.DataFrame, query: str = None, log: bool = False, overwrite: bool = False) -> pd.DataFrame:
        """ Filter spike units based on quality metrics.

        Parameters
        ----------
        quality_metrics : pd.DataFrame
            DataFrame containing the quality metrics for each spike unit.
            This is the dataframe outputted from the spike_unit_quality method
            and will be used to filter spike units.
        query : str
            Query to filter spike units based on the quality metrics.
            This query should be a valid pandas query
        log : bool
            Whether to log the filtered spike units. Default is False.
        overwrite : bool
            Whether to overwrite the existing spike_times dictionary with the filtered units.
            Default is False.

        Returns
        -------
        filtered_units_df : pd.DataFrame
            DataFrame containing the filtered spike units based on the query.

        Notes
        -----
        The quality_metrics dataframe is outputted from the spike_unit_quality method.
        """

        assert isinstance(query, str), f"Query must be a string. Got {type(query)} instead."
        assert isinstance(quality_metrics, pd.DataFrame), "Quality metrics must be a DataFrame. Got {type(quality_metrics)} instead."

        # check if spike units have already been filtered
        if SynaptiConn.spike_unit_filtering:
            if not overwrite:
                msg = ("Spike units have already been filtered. Please re-initialize the object "
                       "or 'set_spike_times' to set the spike_times dict for re-filtering. If this was intentional, "
                       "please set the 'overwrite' parameter to True.")
                warnings.warn(msg)
            if overwrite:
                SynaptiConn.spike_unit_filtering = False

        if not set(self.quality_metric_keys).issubset(quality_metrics.columns):
            msg = ("Quality metrics DataFrame is missing required columns. "
                   f"Required columns: {self.quality_metric_keys}. Please run the spike_unit_quality method.")
            raise DataError(msg)

        # filter units based on query
        filtered_units_df = quality_metrics.query(query)

        # remove filtered units from the spike times dictionary
        self.spike_times = {key: self.spike_times[key] for key in filtered_units_df.index}

        # if log, track removed units
        if log:

            self._setup_log(log_folder_name='removed_spike_units',
                            log_fname='low_quality_units_removed.log')

            removed_units = quality_metrics[~quality_metrics.index.isin(filtered_units_df.index)]

            for key, row in removed_units.iterrows():
                log_msg = f'unit_id: {key} - unit removed from original dataframe with query {query}'
                logging.info(log_msg)

        SynaptiConn.spike_unit_filtering = True

        return filtered_units_df


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


    # def monosynaptic_connection_features(self):
    #     """ Extract connection features from synaptic strength data. """

    #     if hasattr(self, 'pair_synaptic_strength'):

    #     else:
    #         raise ConnectionTypeError("No synaptic strength data found. Please run the synaptic_strength method first.")


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


    def _setup_log(self, log_folder_name: str = 'removed_spike_units',
                   log_fname: str = 'low_quality_units_removed.log'):
        """ Setup logging for specific class methods.

        Parameters
        ----------
        log_folder_name : str
            Name of the log folder to store the log file.
        log_fname : str
            Name of the log file.
        """

        log_folder = pathlib.Path('logs', log_folder_name)
        log_folder.mkdir(parents=True, exist_ok=True)
        log_path = pathlib.Path(log_folder, log_fname).absolute()
        logging.basicConfig(filename=log_path,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            level=logging.INFO,
                            force=True)


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

            # check if spike times need to be converted to milliseconds
            if max_spk_time > self.recording_length_ms:
                self.spike_times[key] = (spks / self.srate) * 1000
                converted_keys.append(key)
            elif max_spk_time > self.recording_length_ms:
                raise SpikeTimesError(f"Spike times for unit {key} exceed the recording length after conversion.")

        if converted_keys:
            converted_keys_str = ', '.join(map(str, converted_keys))
            print(f"Warning: Spike times for unit(s) {converted_keys_str} were converted to milliseconds.")

        SynaptiConn.converted_to_ms = True


    def _check_recording_length_ms(self):
        """ Check the recording length is >= max spike time. """

        for key, spks in self.spike_times.items():
            max_spk_time = np.max(spks)
            if max_spk_time > self.recording_length_ms:
                msg = (f"Spike times for unit {key} exceed the recording length. "
                       f"Max spike time: {max_spk_time}, Recording length: {self.recording_length_ms}. "
                       "Check that the recording length is correct and in milliseconds.")
                raise RecordingLengthError(msg)


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
