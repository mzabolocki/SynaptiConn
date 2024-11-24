""" spike_times.py

Object for handling spike time data.
"""

import warnings
import logging

import numpy as np
import pandas as pd

from synapticonn.utils.errors import SpikeTimesError, DataError, RecordingLengthError, SamplingRateError
from synapticonn.utils.attribute_checks import requires_arguments
from synapticonn.quality_metrics import compute_isi_violations, compute_presence_ratio, compute_firing_rates
from synapticonn.core.core_utils import setup_log
from synapticonn.utils.warnings import custom_formatwarning


###############################################################################
###############################################################################

warnings.formatwarning = custom_formatwarning

###############################################################################
###############################################################################


class SpikeManager():
    """ Base class for checking and managing spike time imports.

    Parameters
    ----------
    spike_times : dict
        Dictionary containing spike times for each unit.
        Indexed by unit ID.
    time_unit : str
        Unit of time for the spike times. Default is 'ms'.
        Options include 's', 'ms'.
    srate : float
        Sampling rate of the recording in Hz.
    recording_length_t : float
        Length of the recording in time units.
    spike_id_type : type
        Type of the spike unit ID. Default is int or str.
    """

    # ----- CLASS VARIABLES
    # quality metric keys used to determine spike unit quality
    quality_metric_keys = ['isi_violations_ratio', 'isi_violations_rate', 'isi_violations_count',
                           'isi_violations_of_total_spikes', 'presence_ratio', 'firing_rate_hz',
                           'recording_length_sec', 'n_spikes']

    # spike unit filtering flag to track if units have been filtered
    spike_unit_filtering = False

    # type of time units allowed
    unit_time_types = ['s', 'ms']


    def __init__(self, spike_times: dict = None,
                 time_unit: str = 'ms',
                 srate: float = None,
                 recording_length_t: float = None,
                 spike_id_type: type = int or str):
        """ Initialize the spike manager. """

        # prepare the spike time data
            # set the spike times, time unit, recording length, and spike ID type
        self.spike_times, self.time_unit, self.recording_length_t, self.spike_id_type, self.srate = \
            self._prepare_spiketime_data(spike_times, time_unit, recording_length_t, spike_id_type, srate)


    def _reset_spike_time_data(self):
        """ Reset the spike times data. """

        self.spike_times = None
        self.recording_length_t = None
        self.srate = None
        self.spike_id_type = None
        self.time_unit = None
        self.spike_unit_filtering = False  # reset the spike unit filtering flag


    def add_spike_time_data(self, spike_times: dict = None,
                            recording_length_t: float = None,
                            time_unit: str = 'ms',
                            srate: float = None,
                            spike_id_type: type = int or str):
        """ Add spike time data to the SpikeManager object.

        Parameters
        ----------
        spike_times : dict
            Dictionary containing spike times for each unit.
            Indexed by unit ID.
        recording_length_t : float
            Length of the recording in time units.
        time_unit : str
            Unit of time for the spike times. Default is 'ms'.
            Options include 's', 'ms'.
        srate : float
            Sampling rate of the recording in Hz.
        spike_id_type : type
            Type of the spike unit ID. Default is int or str.
        """

        # if any data is already present, then clear
        # and the results to ensure object consistency
        self._reset_spike_time_data()

        # prepare the spike time data
        self.spike_times, self.time_unit, self.recording_length_t, self.spike_id_type, self.srate = \
            self._prepare_spiketime_data(spike_times, time_unit, recording_length_t, spike_id_type, srate)


    def report_spike_units(self):
        """ Report the spike units.

        Returns
        -------
        spk_unit_summary : dict
            Dictionary containing the spike unit summary.
            Includes the unit ID, number of spikes, and
            firing rate in Hz.

        Notes
        -----
        The spike unit summary is computed for each unit
        in the spike_times dictionary. Firing rates
        are calculated based on the total number of spikes
        and the recording length. If the time unit is in seconds,
        the firing rate is converted to Hz.
        """

        labels = self.spike_unit_labels()

        # number of spikes for each unit
        n_spks = [len(self.spike_times[label]) for label in labels]

        # calculate firing rates
            # convert to Hz if time unit is in seconds
        if self.time_unit == 's':
            firing_rates = [len(self.spike_times[label]) / self.recording_length_t for label in labels]
        elif self.time_unit == 'ms':
            firing_rates = [len(self.spike_times[label]) / self.recording_length_t * 1000 for label in labels]
        else:
            raise ValueError(f"Invalid time_unit: {self.time_unit}. Must be 's' or 'ms'.")

        spk_unit_summary = {'unit_id': labels,
                            'n_spikes': n_spks,
                            'firing_rate_hz': firing_rates}

        return spk_unit_summary


    def spike_unit_labels(self):
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
        filtered_spike_times : dict
            Filtered dictionary containing spike times for the selected units.

        Notes
        -----
        By default, this method returns all spike times if no units are specified.
        If specific units are specified, then the method will return the spike times
        for those units only.
        """

        if not isinstance(spike_units_to_collect, list):
            raise TypeError("Expected list of spike units. "
                            f"Got {type(spike_units_to_collect)} instead.")

        # return all spike times if no units are specified
        if spike_units_to_collect is None:
            print("No units specified. Returning all spike times.")
            filtered_spike_times = self.spike_times

        # return the selected units
        else:
            try:
                filtered_spike_times = {key: self.spike_times[key] for key in spike_units_to_collect}
            except KeyError as e:
                raise SpikeTimesError(f"Unit {e} not found in the spike times dictionary.")

        return filtered_spike_times


    def spike_unit_quality(self,
                           isi_threshold_t=1.5,
                           min_isi_t=0,
                           presence_ratio_bin_duration_t=60000,
                           presence_ratio_mean_fr_ratio_thresh=0.0) -> pd.DataFrame:
        """ Compute spike isolation quality metrics.

        Parameters
        ----------
        isi_threshold_t : float
            Threshold for the interspike interval (ISI) violations.
        min_isi_t : float
            Minimum ISI value.
        presence_ratio_bin_duration_t : float
            Duration of each bin for the presence ratio.
        presence_ratio_mean_fr_ratio_thresh : float
            Minimum mean firing rate ratio threshold for the presence ratio.
            This is the minimum mean firing rate that must be present in a bin
            for the unit to be considered "present" in that bin.
            By default, this is set to 0.0. This means that the unit must have
            at least one spike in each bin to be considered "present."
        time_unit : str
            Unit of time for the spike times. Default is 'ms'.
            This is used to compute the features in the quality metrics.

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

        These are computed for each spike unit in the spike_times dictionary.

        For further information on the quality metric calculations,
        see the respective functions in the quality_metrics module.
        """

        # TO DO // update this all to account for the time units
        # fix the errors with the calculations also
        # have an option to pass the unit time in the method

        quality_metrics = {}
        for key, spks in self.spike_times.items():

            # isi violations
            isi_violations = compute_isi_violations(spks,
                                                    self.recording_length_t,
                                                    isi_threshold_t,
                                                    min_isi_t)

            # presence ratio
            presence_ratio = compute_presence_ratio(spks,
                                                    self.recording_length_t,
                                                    bin_duration_ms=presence_ratio_bin_duration_t,
                                                    mean_fr_ratio_thresh=presence_ratio_mean_fr_ratio_thresh,
                                                    srate=self.srate)

            # unit firing rates
            firing_rates = compute_firing_rates(spks, self.recording_length_t)

            quality_metrics[key] = isi_violations
            quality_metrics[key].update(presence_ratio)
            quality_metrics[key].update(firing_rates)

        return pd.DataFrame(quality_metrics).T


    def filter_spike_units(self, quality_metrics: pd.DataFrame,
                           query: str = None,
                           log: bool = False,
                           overwrite: bool = False) -> pd.DataFrame:
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
        if SpikeManager.spike_unit_filtering:
            if not overwrite:
                msg = ("Spike units have already been filtered. Please re-initialize the object "
                       "or 'set_spike_times' to set the spike_times dict for re-filtering. "
                       "If this was intentional, please set the 'overwrite' parameter to True.")
                warnings.warn(msg)
            if overwrite:
                SpikeManager.spike_unit_filtering = False

        if not set(self.quality_metric_keys).issubset(quality_metrics.columns):
            msg = ("Quality metrics DataFrame is missing required columns. "
                   f"Required columns: {self.quality_metric_keys}. "
                   "Please run the spike_unit_quality method.")
            raise DataError(msg)

        # filter units based on query
        filtered_units_df = quality_metrics.query(query)

        # remove filtered units from the spike times dictionary
        self.spike_times = {key: self.spike_times[key] for key in filtered_units_df.index}

        # if log, track removed units
        if log:

            setup_log(log_folder_name='removed_spike_units',
                      log_fname='low_quality_units_removed.log')

            removed_units = quality_metrics[~quality_metrics.index.isin(filtered_units_df.index)]

            for key, row in removed_units.iterrows():
                log_msg = f'unit_id: {key} - unit removed from original dataframe with query {query}'
                logging.info(log_msg)

        SpikeManager.spike_unit_filtering = True

        return filtered_units_df


    @requires_arguments('spike_id_type','srate', 'spike_times',
                        'recording_length_t', 'time_unit')
    def _prepare_spiketime_data(self, spike_times: dict = None,
                                time_unit: str = None,
                                recording_length_t: float = None,
                                spike_id_type: type = None,
                                srate: type = float or int):
        """ Prepare the spike time data.

        Parameters
        ----------
        spike_times : dict
            Dictionary containing spike times for each unit.
            Indexed by unit ID.
        srate : float or int
            Sampling rate of the recording in Hz.
        time_unit : str
            Unit of time for the spike times. Default is 'ms'.
            Options include 's', 'ms'.
        recording_length_t : float
            Length of the recording in time units.
        spike_id_type : type
            Type of the spike unit ID. Default is int or str.

        Returns
        -------
        Returns the validated spike times, time unit, recording length, and spike ID type.
        """

        ## sampling rate checks - run these checks on the sampling rate input

        if not isinstance(srate, (float, int)):
            raise SamplingRateError("Sampling rate must be a float or int.")
        if srate <= 0:
            raise SamplingRateError("Sampling rate must be greater than 0.")
        if srate < 5000 or srate > 30_000:
            warnings.warn("Sampling rate is outside the typical range of 5000-30,000 Hz. "
                          "Please verify the sampling rate used to perform spike unit extractions. "
                          "Please taken into account the Nyquist frequency when analyzing the data. "
                          "If this is intentional, please ignore this warning.")

        ## spike time checks - run these checks on the spike times dict input

        # check that spike_times is a dictionary
            # see documentation for more information on the expected format
        if not isinstance(spike_times, dict):
            raise SpikeTimesError("Spike times must be a dictionary. Each key should be the unit ID.")

        # check that all keys in spike_times match the expected type
        if not all(isinstance(key, spike_id_type) for key in spike_times.keys()):
            raise SpikeTimesError(
                f"All keys in spike_times must be of type '{spike_id_type.__name__}'. "
                "Please verify the unit IDs or update the spike_id_type."
            )

        # validate the spike times for each unit
        for unit_id, spks in spike_times.items():
            # check if spike times array is empty
            if len(spks) == 0:
                raise SpikeTimesError(f"Spike times for unit {unit_id} are empty. Please check the data.")

            # check for non-negative spike times
            if not np.all(spks >= 0):
                raise SpikeTimesError(f"Spike times for unit {unit_id} must be non-negative.")

            # check that each spike time is a float
            if not np.issubdtype(spks.dtype, np.floating):
                raise SpikeTimesError(
                    f"Spike times for unit {unit_id} must be of type 'float'. "
                    f"Found type '{spks.dtype}' instead."
                )

            # check if any nan or inf values in spike times
            if np.any(np.isnan(spks)) or np.any(np.isinf(spks)):
                raise SpikeTimesError(f"Spike times for unit {unit_id} contain NaN or Inf values.")

        # check the time unit type
        time_unit = self._time_unit_check(time_unit)

        # ensure spike times do not exceed recording length
            # if so, probably an error in the recording length, spike times or time unit
        for unit_id, spks in spike_times.items():
            if np.max(spks) > recording_length_t:
                msg = (f"Spike times for unit {unit_id} exceed the recording length. "
                       f"Max spike time: {np.max(spks)}, Recording length: {recording_length_t}. "
                       f"Check that the recording length is correct and in {time_unit}.")
                raise RecordingLengthError(msg)

        # check the time unit matches the recording length
            # raise a warning if the spike times exceed the recording length
        max_spk_time = np.max([np.max(spks) for spks in spike_times.values()])
        if max_spk_time > 0.8*recording_length_t:
            msg = ("Unit is firing across less than 20% of the recording length. "
                   "This may lead to unexpected results. Please check the spike times and recording length "
                   "passed to the SpikeManager object. If this is intentional, please ignore this warning.")
            warnings.warn(msg)

        return spike_times, time_unit, recording_length_t, spike_id_type, srate


    def _time_unit_check(self, time_unit: str = None):
        """ Check the time unit. """

        if time_unit not in self.unit_time_types:
            raise TypeError(
                f"Time unit must be in {self.unit_time_types}. "
                f"Got {time_unit} instead."
                )

        return time_unit