""" connections.py

Base model object, which is used to quantify monosynaptic connections between neurons.
"""

import warnings
import logging

import pandas as pd

from typing import Any, List, Tuple

from synapticonn.core.spike_times import SpikeManager
from synapticonn.plots import plot_acg, plot_ccg, plot_ccg_synaptic_strength, plot_spiketrain
from synapticonn.monosynaptic_connections.ccg_synaptic_strength import calculate_synaptic_strength
from synapticonn.monosynaptic_connections.ccg_connection_type import get_putative_connection_type
from synapticonn.postprocessing.crosscorrelograms import compute_crosscorrelogram
from synapticonn.features import compute_peak_latency, compute_ccg_bootstrap, compute_ccg_cv, compute_peak_amp
from synapticonn.core.core_utils import _validate_spike_pairs, _validate_parameter, setup_log
from synapticonn.core.info import get_available_processing_methods
from synapticonn.utils.errors import SpikeTimesError, DataError
from synapticonn.utils.attribute_checks import requires_arguments
from synapticonn.utils.report import gen_model_results_str
from synapticonn.utils.warnings import custom_formatwarning
from synapticonn.utils.data_utils import flatten_list


###############################################################################
###############################################################################

warnings.formatwarning = custom_formatwarning

###############################################################################
###############################################################################


class SynaptiConn(SpikeManager):
    """
    Base class for quantifying monosynaptic connections between neurons.

    This class builds upon the `SpikeManager` to provide functionality for 
    computing, analyzing, and visualizing monosynaptic connections between 
    pairs of neurons based on their spike trains. Key features include 
    cross-correlogram computation, synaptic strength estimation, and connection 
    classification.

    Parameters
    ----------
    spike_times : dict
        Dictionary containing spike times for each unit, indexed by unit ID. Spike 
        times should be a NumPy array of floats.
    time_unit : str
        Unit of time for the spike times. Options are 'ms' (milliseconds) or 's' (seconds).
    bin_size_t : float
        Bin size for computing cross-correlograms.
    max_lag_t : float
        Maximum lag to consider when computing cross-correlograms.
    method : str, optional
        Method for computing synaptic strength. Default is 'cross-correlation'. 
        Currently, only this method is implemented, but future versions may 
        include additional methods.
    recording_length_t : float
        Duration of the recording, in the same time unit as `time_unit`.
    srate : float
        Sampling rate of the recording in Hz.
    spike_id_type : type
        Data type of spike unit identifiers. Typically `int` or `str`.

    Attributes
    ----------
    bin_size_t : float
        Bin size used for cross-correlogram computations.
    max_lag_t : float
        Maximum lag used for cross-correlogram computations.
    method : str
        The method used for synaptic strength computation.
    recording_length_t : float
        Total duration of the recording, in the specified time unit.
    srate : float
        Sampling rate of the spike data, in Hz.
    spike_id_type : type
        Type of spike unit identifiers (e.g., `int` or `str`).
    time_unit : str
        Unit of time used for computations and spike data.
    pair_synaptic_strength : dict
        Dictionary containing synaptic strength data for each spike pair. This 
        is populated after running the `fit` or `synaptic_strength` methods.

    Notes
    -----
    - The `SynaptiConn` class is designed to process and analyze spike train data 
      for determining monosynaptic connections.
    - If the spike times are provided in seconds, they are converted to milliseconds 
      internally to maintain consistency.
    - The class assumes that the recording duration (`recording_length_t`) and 
      time unit (`time_unit`) are accurately specified.

    Examples
    --------
    Initialize the SynaptiConn object and compute synaptic strength:

    >>> synapti_conn = SynaptiConn(
    ...     spike_times=spike_data,
    ...     time_unit='ms',
    ...     bin_size_t=1.0,
    ...     max_lag_t=50.0,
    ...     method='cross-correlation',
    ...     recording_length_t=60000,
    ...     srate=20000
    ... )
    >>> synapti_conn.fit(spike_pairs=[(1, 2), (3, 4)])
    >>> print(synapti_conn.pair_synaptic_strength)
    """

    # connection filtering flag to track if spike pairs have been filtered
    connections_filtering = False


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

        self.bin_size_t = self._bin_size_check(bin_size_t, max_lag_t)
        self.max_lag_t = self._max_lag_check(bin_size_t, max_lag_t)
        self.method = self._method_check(method)


    def __call__(self):
        """ Return the object. """

        return self


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
            'spike_id_type': self.spike_id_type,
            'time_unit': self.time_unit
            }

        if self.method == 'cross-correlation':  # default settings for the cross-correlation method

            crosscorr_connection_settings = {
                'bin_size_t': 1,
                'max_lag_t': 100,
                'num_iterations': 1000,
                'jitter_range_t': 10,
                'half_window_t': 5,
                'time_unit': 'ms',
                'n_jobs': -1
            }

            settings.update(crosscorr_connection_settings)

            return settings

        else:
            raise NotImplementedError("Only the 'cross-correlation' method is currently implemented.")


    def spike_unit_ids(self):
        """ Return the spike unit labels. """

        return list(self.spike_times.keys())


    @requires_arguments('bin_size_t', 'max_lag_t', 'time_unit')
    def set_bin_settings(self, bin_size_t: float = None,
                         max_lag_t: float = None,
                         time_unit: str = None,
                         verbose: bool = True):
        """ Set the settings of the object.

        Useful for changing the bin size and maximum lag after initialization.

        Parameters
        ----------
        bin_size_t : float
            Bin size of the cross-correlogram or auto-correlograms.
        max_lag_t : float
            Maximum lag to compute the cross-correlogram.
        time_unit : str
            Time unit options in ms (milliseconds) or s (seconds).
            These are used to set the time unit for the spike times, recording length, 
            bin size, and maximum lag for all processing.
        verbose : bool
            If True, print a message confirming the settings.

        User Warning
        ------------
        Please use this method with caution. Parameters should match the spike time units.
        If a change in the time unit is required, the spike times will NOT be converted to the new unit
        automatically. This should be done manually before calling this method.
        """

        warnings.warn("This method is used to set the bin size and maximum "
                      "lag after initialization. Please use this method with caution. "
                      "Parameters should match the spike time units", UserWarning)

        self.time_unit = self._time_unit_check(time_unit)
        self.bin_size_t = self._bin_size_check(bin_size_t, max_lag_t)
        self.max_lag_t = self._max_lag_check(bin_size_t, max_lag_t)

        if verbose:
            print(f"Bin size set to {self.bin_size_t} {self.time_unit}, "
                  f"and maximum lag set to {self.max_lag_t} {self.time_unit}.")


    @requires_arguments('time_unit')
    def set_time_unit(self, time_unit: str = 'ms'):
        """ Set the time unit for the spike times, recording length, bin size, and maximum lag.

        Parameters
        ----------
        time_unit : str
            Time unit options in ms (milliseconds) or s (seconds).
        """

        warnings.warn("This method is used to set the time unit for the spike times. "
                      "All processing will be based on this time unit.", UserWarning)

        self.time_unit = self._time_unit_check(time_unit)


    def reset_pair_synaptic_strength(self):
        """ Reset the synaptic strength data. """

        if hasattr(self, 'pair_synaptic_strength'):
            del self.pair_synaptic_strength
        else:
            raise DataError("No synaptic strength data found.")


    @requires_arguments('spike_pairs', 'synaptic_strength_threshold')
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
            - jitter_range_t : float
                Jitter range for synaptic strength computation (in ms, default: 10.0).
            - half_window_t : float
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
            - jitter_range_t : float
                Jitter range for synaptic strength computation (in ms, default: 10.0).
            - half_window_t : float
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
            This is stored in the object for future reference, and can be accessedusing the 'pair_synaptic_strength' attribute.
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


    def print_connection_results(self, connection_types: dict = None,
                                 concise: bool = False,
                                 params: dict = None):
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


    @requires_arguments('spike_pairs', 'num_iterations',
                        'max_lag_t', 'bin_size_t',
                        'jitter_range_t', 'half_window_t')
    def synaptic_strength(self,
                          spike_pairs: List[Tuple] = None,
                          num_iterations: int = 1000,
                          max_lag_t: float = 25.0,
                          bin_size_t: float = 0.5,
                          jitter_range_t: float = 10.0,
                          half_window_t: float = 5,
                          n_jobs: int = -1) -> dict:
        """ Compute the synaptic strength for the given spike pairs.

        Parameters
        ----------
        spike_unit_ids : list
            List of spike unit labels.
        spike_pairs : List[Tuple]
            List of spike pairs to compute synaptic strength.
            These are tuples of pre- and post-synaptic neuron IDs.
            Pre-synaptic neuron ID is the first element and post-synaptic neuron ID is the second element.
        num_iterations : int
            Number of iterations to compute the synaptic strength.
        max_lag_t : float
            Maximum lag to compute the synaptic strength.
        bin_size_t : float
            Bin size of the synaptic strength.
        jitter_range_t : float
            Jitter range to compute the synaptic strength.
        half_window_t : float
            Half window size for the synaptic strength.
        n_jobs: int
            Number of parallel jobs to run. Default is -1.
            Use this to speed up computation.

        Returns
        -------
        synaptic_strength_pairs : dict
            Dictionary containing synaptic strength data for all pairs of spike trains.
            This contains the mean, standard deviation, and confidence intervals of the synaptic strength
            following jittering and bootstrapping.

        Warning
        -------
        If spike times are not in milliseconds, a DataError is raised. Please convert to milliseconds
        for synaptic strength calculations using the 'cross-correlation' method. This is based on [1].

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

        # check spike time units
            # if not in milliseconds, raise an error
            # this is required for synaptic strength calculations
            # please see [1] for more details
        if self.time_unit == 's':
            raise DataError("Spike times are not in milliseconds. Please convert to milliseconds "
                            f"for synaptic strength calculations using {self.method}. "
                            "This can be done by setting the time unit to 'ms' using "
                            "the set_time_unit method.")

        # get spike unit ids
        spike_unit_ids = self.spike_unit_ids()

        # filter passed spike pairs for available spike units
        valid_spike_pairs = _validate_spike_pairs(spike_pairs, spike_unit_ids)

        self.pair_synaptic_strength = {}
        for pre_synaptic_neuron_id, post_synaptic_neuron_id in valid_spike_pairs:

            # retrieve spike times for the pre- and post-synaptic neurons
            pre_synaptic_spktimes = self.get_spike_times_for_units([pre_synaptic_neuron_id]).get(pre_synaptic_neuron_id)
            post_synaptic_spktimes = self.get_spike_times_for_units([post_synaptic_neuron_id]).get(post_synaptic_neuron_id)

            # calculate synaptic strength
            synaptic_strength_data = calculate_synaptic_strength(pre_synaptic_spktimes,
                                                                 post_synaptic_spktimes,
                                                                 jitter_range_t=jitter_range_t,
                                                                 num_iterations=num_iterations,
                                                                 max_lag_t=max_lag_t,
                                                                 bin_size_t=bin_size_t,
                                                                 half_window_t=half_window_t,
                                                                 n_jobs=n_jobs)

            self.pair_synaptic_strength[(pre_synaptic_neuron_id, post_synaptic_neuron_id)] = synaptic_strength_data

        return self.pair_synaptic_strength


    def monosynaptic_connection_types(self, synaptic_strength_threshold: float = None) -> dict:
        """ Categorize monosynaptic connection types based on synaptic strength data output.

        Parameters
        ----------
        synaptic_strength_threshold : float
            Threshold value for categorizing connection types.
            Default is None. This is used to categorize the
            connection types based on the synaptic strength values.

        Returns
        -------
        connection_types : dict
            Dictionary containing connection types for all pairs of spike trains.

        Cross-correlation method notes:
        ------------------------------
        Based on [1], for excitatory connections, a threshold of  5 is recommended.
        For inhibitory connections, a threshold of  -5 is recommended. Thresholds
        can be adjusted based on the synaptic strength data.

        Please see [1] for more details.

        Notes
        -----
        Currently, connection types are based on the synaptic strength values. This is 
        computed using the cross-correlation method. In future versions, this will be expanded
        to include other types of correlation methods.

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


    def filter_connections(self, connections_df: pd.DataFrame,
                           query: str = None,
                           log: bool = False,
                           overwrite: bool = False) -> pd.DataFrame:
        """ Filter connections based on a query string.

        Parameters
        ----------
        connections_df : pd.DataFrame
            DataFrame containing the connection data.
            This DataFrame should contain the synaptic strength data
            for all select pairs of spike trains.
        query : str
            Query string to filter the DataFrame.
        log : bool
            If True, print the query string used for filtering.
        overwrite : bool
            If True, overwrite the DataFrame with the filtered data.

        Returns
        -------
        connections_df_filtered : pd.DataFrame
            DataFrame containing the filtered connection data.

        Log
        ---
        The log parameter is used to track the removed spike pairs with
        detected monosynaptic connections. This is useful for tracking
        the removed spike pairs from the original DataFrame.

        The output is saved in a log file in the
        'removed_spikepair_connections' folder. The file is named
        'low_quality_connections_removed.log'.
        """

        # validate the input data
        if connections_df is None:
            raise DataError("No connection data found. "
                            "Please run the synaptic_strength method first.")

        # check types
        if not isinstance(connections_df, pd.DataFrame):
            raise TypeError(f"Connections data must be a pandas DataFrame. \
                            Got {type(connections_df)} instead.")
        if not isinstance(query, str):
            raise TypeError(f"Query must be a string. Got {type(query)} instead.")

        # check if the connection data is already filtered
        if SynaptiConn.connections_filtering:
            if not overwrite:
                msg = ("Connections have already been filtered. Please re-initialize the object. "
                       "If this was intentional, please set the 'overwrite' parameter to True.")
                warnings.warn(msg)
            if overwrite:
                SynaptiConn.connections_filtering = False

        # filter based on the query
        if query is not None:
            connections_df_filtered = connections_df.query(query)
        else:
            # skip filtering if no query is provided
            connections_df_filtered = connections_df

        # if log, track removed spike pairs
        # with detected monosynaptic connections
        if log:

            setup_log(log_folder_name='removed_spikepair_connections',
                      log_fname='low_quality_connections_removed.log')

            removed_spikepairs = connections_df[~connections_df.index.isin(connections_df_filtered.index)]

            for key, row in removed_spikepairs.iterrows():
                log_msg = f'spike pair: {key} - removed from original dataframe with query {query}'
                logging.info(log_msg)

        SynaptiConn.connections_filtering = True

        return connections_df_filtered


    def return_crosscorrelogram_data(self, spike_pairs: List[Tuple] = None) -> dict:
        """ Compute and return the cross-correlogram data for valid spike pairs.

        Parameters
        ----------
        spike_pairs : List[Tuple]
            List of spike pairs to compute the cross-correlogram data.

        Returns
        -------
        crosscorrelogram_data : dict
            Dictionary containing cross-correlograms and bins for all pairs of spike trains.
        """

        if self.time_unit == 's':
            warnings.warn("Spike times are not in milliseconds. It is recommended to convert to milliseconds "
                          "for cross-correlogram calculations.", UserWarning)

        # validate spike pairs
        valid_spike_pairs = _validate_spike_pairs(spike_pairs, self.spike_unit_ids())

        # unpack the spike pairs to find spike IDs
        spike_ids = flatten_list(valid_spike_pairs)
        # and retrieve spike times and compute cross-correlogram data
        spike_times = self.get_spike_times_for_units(spike_ids)

        crosscorrelogram_data = compute_crosscorrelogram(
            spike_times, valid_spike_pairs, bin_size_t=self.bin_size_t, max_lag_t=self.max_lag_t)

        return crosscorrelogram_data


    def plot_synaptic_strength(self, spike_pair: tuple = None, **kwargs):
        """ Plot the synaptic strength for the given spike pair.

        Note, this method requires the synaptic strength data to be computed first.
        It only plots the synaptic strength for a single pair of spike trains.

        Parameters
        ----------
        spike_pair : tuple
            Spike pair to plot the synaptic strength.
        **kwargs
            Additional keyword arguments passed to `plot_ccg_synaptic_strength`.

        Raises
        ------
        DataError
            If no synaptic strength data is found.
        SpikePairError
            If spike pair is not provided.
        NotImplementedError
            If the method is not implemented in the current package version.
        """

        if not hasattr(self, 'pair_synaptic_strength'):
            raise DataError("No synaptic strength data found. Please run "
                            "the synaptic_strength method first.")

        # validate spike pair selection
        valid_spike_pair = _validate_spike_pairs([spike_pair], self.spike_unit_ids())

        # check if the method is implemented
        # note that only the 'cross-correlation' method is currently implemented
        # and for future versions, this will be expanded to include other types of correlation methods
        if self.method == 'cross-correlation':
            plot_ccg_synaptic_strength(self.pair_synaptic_strength, valid_spike_pair[0], self.time_unit, **kwargs)
        else:
            raise NotImplementedError("Only the 'cross-correlation' method is currently"
                                      " implemented for plot. Please choose this method.")


    def plot_autocorrelogram(self, spike_units: list = None, **kwargs):
        """ Plot the autocorrelogram.

        Parameters
        ----------
        spike_units : list
            List of spike units to plot.
        **kwargs
            Additional keyword arguments passed to `plot_acg`.

        Notes
        -----
        Autocorrelograms are computed for each spike unit and plotted.
        The bin size and maximum lag are set by the object parameters.
        """

        spike_times = self.get_spike_times_for_units(spike_units)

        if len(spike_times) == 0:
            raise SpikeTimesError("No valid spike units to plot.")

        # plot the autocorrelogram
        plot_acg(spike_times,
                 bin_size_t=self.bin_size_t,
                 max_lag_t=self.max_lag_t,
                 time_unit=self.time_unit,
                 **kwargs)


    def plot_crosscorrelogram(self, spike_pairs: List[Tuple] = None, **kwargs: Any):
        """ Plot the cross-correlogram for valid spike pairs.

        Parameters
        ----------
        spike_pairs : List[Tuple]
            List of spike pairs to plot.
        ax : Any
            Axis to plot on.
        show_axes : bool
            Whether to add axis labels. Default is True.
        **kwargs : Any
            Additional keyword arguments passed to `plot_ccg`.
        """

        crosscorrelogram_data = self.return_crosscorrelogram_data(spike_pairs)
        plot_ccg(crosscorrelogram_data, time_unit=self.time_unit, **kwargs)


    def plot_spike_train(self, spike_units: list = None, **kwargs: Any):
        """ Plot the spike train for the given spike units.

        Parameters
        ----------
        spike_units : list
            List of spike units to plot.
            If None, all spike units are plotted.
        **kwargs : Any
            Additional keyword arguments passed to `plot_spike_train`
            for customizing the plot using `matplotlib`.
        """

        if spike_units is None:
            spktimes_plotting = self.spike_times  # default to all spike times

        elif spike_units is not None:
            try: 
                spktimes_plotting = {k: self.spike_times[k] for k in spike_units}
            except KeyError as e:
                raise SpikeTimesError(f"Please provide valid spike unit keys to plot. "
                                      f"Error with spike units: {e}")

        plot_spiketrain(spktimes_plotting, time_unit=self.time_unit, **kwargs)


    def _bin_size_check(self, bin_size_t, max_lag_t):
        """ Check if the bin size is valid. """

        _validate_parameter(
            name="Bin size",
            value=bin_size_t,
            min_value=0,
            max_value=min(self.recording_length_t, max_lag_t),
            warn_threshold=0.001 if self.time_unit == 's' else 1,
            warn_message="Bin size is greater than 1 ms. This may lead to inaccurate results.",
        )
        return bin_size_t


    def _max_lag_check(self, bin_size_t, max_lag_t):
        """ Check if the maximum lag is valid. """

        _validate_parameter(
            name="Maximum lag",
            value=max_lag_t,
            min_value=0,
            max_value=self.recording_length_t,
        )
        if max_lag_t < bin_size_t:
            raise ValueError("Maximum lag must be greater than or equal to the bin size.")
        return max_lag_t


    def _method_check(self, method):
        """ Check if the method is valid. """

        if method not in get_available_processing_methods():
            raise NotImplementedError(f"Method {method} is not implemented. "
                                      f"Please choose from {get_available_processing_methods()}.")
        return method
