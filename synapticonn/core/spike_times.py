""" Object for handling spike time data. """

import warnings
import logging
import pathlib

import numpy as np
import pandas as pd

from synapticonn.utils.errors import SpikeTimesError, DataError, RecordingLengthError
from synapticonn.utils.attribute_checks import requires_sampling_rate, requires_recording_length
from synapticonn.quality_metrics import compute_isi_violations, compute_presence_ratio, compute_firing_rates
from synapticonn.core.core_tools import setup_log


###############################################################################
###############################################################################


class SpikeManager():
    """ Base class for checking and managing spike time imports. """

    # ----- CLASS VARIABLES
    # spike unit filtering
    spike_unit_filtering = False

    # quality metric keys
    quality_metric_keys = ['isi_violations_ratio', 'isi_violations_rate', 'isi_violations_count',
                           'isi_violations_of_total_spikes', 'presence_ratio', 'firing_rate_hz',
                           'recording_length_sec', 'n_spikes']


    def __init__(self, spike_times: dict = None,
                 time_unit: str = 'ms',
                 srate: float = None,
                 recording_length_t: float = None,
                 spike_id_type: type = int or str):
        """ Initialize the spike manager. """

        self.spike_times = spike_times or {}
        self.srate = srate
        self.recording_length_t = recording_length_t
        self.spike_id_type = spike_id_type
        self.time_unit = time_unit

        self._run_time_unit_checks()
        self._run_initial_spike_time_type_checks()
        self._run_initial_spike_time_val_checks()


    def report_spike_units(self):
        """ Report the spike units. """

        labels = self.spike_unit_labels()
        n_spks = [len(self.spike_times[label]) for label in labels]
        firing_rates = [len(self.spike_times[label]) / self.recording_length_t*1000 for label in labels]
        spk_unit_summary = {'unit_id': labels, 'n_spikes': n_spks, 'firing_rate_hz': firing_rates}

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

        These are computed for each spike unit in the spike_times dictionary.

        For further information on the quality metric calculations,
        see the respective functions in the quality_metrics module.
        """

        quality_metrics = {}
        for key, spks in self.spike_times.items():

            # isi violations
            isi_violations = compute_isi_violations(spks, self.recording_length_t,
                                                    isi_threshold_ms, min_isi_ms)

            # presence ratio
            presence_ratio = compute_presence_ratio(spks, self.recording_length_t,
                                                    bin_duration_ms=presence_ratio_bin_duration_ms,
                                                    mean_fr_ratio_thresh=presence_ratio_mean_fr_ratio_thresh,
                                                    srate=self.srate)

            # unit firing rates
            firing_rates = compute_firing_rates(spks, self.recording_length_t)

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
        if SpikeManager.spike_unit_filtering:
            if not overwrite:
                msg = ("Spike units have already been filtered. Please re-initialize the object "
                       "or 'set_spike_times' to set the spike_times dict for re-filtering. If this was intentional, "
                       "please set the 'overwrite' parameter to True.")
                warnings.warn(msg)
            if overwrite:
                SpikeManager.spike_unit_filtering = False

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

            setup_log(log_folder_name='removed_spike_units',
                      log_fname='low_quality_units_removed.log')

            removed_units = quality_metrics[~quality_metrics.index.isin(filtered_units_df.index)]

            for key, row in removed_units.iterrows():
                log_msg = f'unit_id: {key} - unit removed from original dataframe with query {query}'
                logging.info(log_msg)

        SpikeManager.spike_unit_filtering = True

        return filtered_units_df

    ########### TIME UNIT CHECKS ###########

    def _run_time_unit_checks(self):
        """ Run the time unit checks at initialization. """

        unit_times = ['ms', 's']

        if self.time_unit not in unit_times:
            raise ValueError(f"Time unit must be in {unit_times}. Got {self.time_unit} instead.")


    ########### SPIKE TIME TYPE CHECKS ###########


    def _run_initial_spike_time_type_checks(self):
        """ Run all the spike-time-related type checks at initialization. """

        self._check_spike_times_type()
        self._check_key_types(self.spike_times, self.spike_id_type)


    def _check_spike_times_type(self):
        """ Ensure spike times is a dictionary. """

        if not isinstance(self.spike_times, dict):
            raise SpikeTimesError('Spike times must be a dictionary with unit-ids as keys.')


    def _check_key_types(self, d, key_type):
        """ Validate that all keys in a dictionary are of a specified type.

        Parameters:
        ----------
        d : dict
            Dictionary to check.
        key_type : type
            Type that all keys must be.

        Raises:
        -------
        ValueError
            If any key in the dictionary is not of the specified type.
        """

        if not all(isinstance(key, key_type) for key in d.keys()):
            raise SpikeTimesError(f"All keys must be of type {key_type.__name__}."
                                  " Please check the spike unit IDs and change accordingly, or change the spike_id_type.")


    ########### SPIKE TIME VALUE CHECKS ###########


    def _run_initial_spike_time_val_checks(self):
        """ Run all the spike-time-related value checks at initialization. """

        self._check_spike_time_conversion()
        self._check_negative_spike_times()
        self._check_spike_times_type()
        self._check_recording_length_t()


    @requires_sampling_rate
    @requires_recording_length
    def _check_spike_time_conversion(self):
        """ Check that spike time values are in millisecond format. """

        try:
            # check for spike times type incase it was changed
            # with object re-initialization
            self._check_spike_times_type()
        except SpikeTimesError:
            pass

        for unit_id, spks in self.spike_times.items():
            if len(spks) == 0:
                raise SpikeTimesError(f"Spike times for unit {unit_id} are empty.")

            max_spk_time = np.max(spks)

            # check if spike times exceed recording length
            if max_spk_time > self.recording_length_t:
                raise SpikeTimesError(f"Spike times for unit {unit_id} exceed the recording length after conversion.",
                                      f" Check that the recording length is correct and in the time unit type: {self.time_unit}.")


    def _check_recording_length_t(self):
        """ Check the recording length is >= max spike time. """

        for unit_id, spks in self.spike_times.items():
            max_spk_time = np.max(spks)
            if max_spk_time > self.recording_length_t:
                msg = (f"Spike times for unit {unit_id} exceed the recording length. "
                       f"Max spike time: {max_spk_time}, Recording length: {self.recording_length_t}. "
                       f"Check that the recording length is correct and in {self.time_unit}.")
                raise RecordingLengthError(msg)


    def _check_negative_spike_times(self):
        """ Check for negative spike times. """

        for unit_id, spks in self.spike_times.items():
            if not np.all(spks >= 0):
                raise SpikeTimesError(f'Spike times for unit {unit_id} must be non-negative.')


    def _check_spike_times_type(self):
        """ Check the type of the spike times dictionary are in floats. """

        for unit_id, spks in self.spike_times.items():

            # check the type of individual spike times
            for spk_count, spk in enumerate(spks):
                if not isinstance(spk, np.floating):
                    raise SpikeTimesError(f'Spike times for unit {unit_id} at {spk_count} element must be a float type.',
                                          f' Got {type(spk)} instead.')