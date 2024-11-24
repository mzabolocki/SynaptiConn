""" info.py

Internal functions to manage info related to the Synapticonn core.
"""


def get_attribute_description():
    """
    Return descriptions of various attributes used in the synaptic connection and spike time analysis.

    Returns
    -------
    attributes : dict
        A dictionary containing descriptions of quality control metrics,
        processing methods, and additional attributes.
    """
    attributes = {
        'qc_metrics': {
            'isi_violations_ratio': (
                'The ratio of the number of interspike interval (ISI) '
                'violations to the total number of spikes.'
            ),
            'isi_violations_rate': 'The rate of interspike interval (ISI) violations.',
            'isi_violations_count': 'The number of interspike interval (ISI) violations.',
            'isi_violations_of_total_spikes': (
                'The ratio of the number of interspike interval (ISI) '
                'violations to the total number of spikes.'
            ),
            'presence_ratio': (
                'The fraction of time during a recording that a unit is spiking '
                'above a defined threshold.'
            ),
            'firing_rate_hz': 'The mean firing rate in Hz (spikes per second).',
            'recording_length_sec': 'The total duration of the recording in seconds.',
            'n_spikes': 'The total number of spikes recorded for a unit.',
        },
        'processing_methods': {
            'cross-correlation': (
                'A measure of similarity between two signals. '
                'Used to identify synaptic connections based on spike train patterns.'
            ),
            'auto-correlation': (
                'A measure of similarity of a signal with a delayed version of itself. '
                'Used to analyze patterns within a single spike train.'
            ),
            'jittering_analysis': (
                'A statistical approach to assess synaptic strength by introducing '
                'controlled temporal perturbations to spike times.'
            ),
            'peak_detection': (
                'Identifies significant peaks in cross-correlograms to estimate '
                'synaptic strength or latency.'
            ),
        },
        'spike_time_attributes': {
            'spike_times': (
                'A dictionary containing spike times for each unit, indexed by unit IDs.'
            ),
            'spike_unit_labels': 'A list of labels representing each spike unit.',
            'time_unit': (
                "The unit of time for spike data. Can be 'ms' (milliseconds) or 's' (seconds)."
            ),
            'recording_length_t': 'The total duration of the recording in the given time unit.',
            'spike_id_type': 'The data type of spike unit identifiers (e.g., int or str).',
            'sampling_rate': 'The frequency at which the recording was sampled, in Hz.',
        },
        'connection_attributes': {
            'synaptic_strength': (
                'A measure of the strength of a monosynaptic connection between two neurons, '
                'typically derived from cross-correlograms.'
            ),
            'connection_types': (
                'Classification of synaptic connections as excitatory, inhibitory, or non-significant '
                'based on thresholds applied to synaptic strength data.'
            ),
            'peak_latency': (
                'The time difference at which the peak of a cross-correlogram occurs, '
                'indicating the latency of a synaptic connection.'
            ),
            'crosscorrelogram': (
                'A histogram of time differences between spikes from two units, '
                'used to analyze their interaction and connection type.'
            ),
        },
        'miscellaneous': {
            'bin_size_t': (
                'The time bin size (in milliseconds) used for computing cross-correlograms or auto-correlograms.'
            ),
            'max_lag_t': (
                'The maximum lag time (in milliseconds) considered when computing cross-correlograms.'
            ),
            'num_iterations': (
                'The number of iterations for statistical analyses like bootstrapping or jittering.'
            ),
            'jitter_range_t': (
                'The range of temporal jitter (in milliseconds) applied to spike trains for analysis.'
            ),
        },
    }
    return attributes


def get_quality_metric_keys():
    """ Return the quality metric keys.

    Returns
    -------
    quality_metric_keys : list
        List of quality metric keys. These 
        are the keys that are used to store
        the quality metrics in the quality
        metric dictionary.
    """

    quality_metric_keys = ['isi_violations_ratio',
                           'isi_violations_rate',
                           'isi_violations_count',
                           'isi_violations_of_total_spikes',
                           'presence_ratio',
                           'firing_rate_hz',
                           'recording_length_sec',
                           'n_spikes']

    return quality_metric_keys


def get_available_processing_methods():
    """ Return the available processing methods.

    Returns
    -------
    available_processing_methods : list
        List of available processing methods.
        These are the methods that can be used
        to process the data in the Synapticonn
        core.
    """

    available_processing_methods = ['cross-correlation']

    return available_processing_methods


def get_unit_time_types():
    """ Return the unit time types.

    Returns
    -------
    unit_time_types : list
        List of unit time types. These are
        the types of time units that are
        allowed in the Synapticonn core.
    """

    unit_time_types = ['s', 'ms']

    return unit_time_types
