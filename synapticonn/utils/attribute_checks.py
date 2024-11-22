""" attribute_checks.py

Utils checking attributes of an object if present or not.
"""

from ..utils.errors import NoDataError


#######################################################
#######################################################


def requires_sampling_rate(func):
    """ Decorator to ensure that 'srate' (sampling rate) is provided in the instance. """

    def wrapper(self, *args, **kwargs):
        if getattr(self, 'srate', None) is None:
            raise NoDataError('The sampling rate (srate) must be provided.')

        return func(self, *args, **kwargs)

    return wrapper


def requires_recording_length(func):
    """ Decorator to ensure that 'recording_length_t' is provided in the instance. """

    def wrapper(self, *args, **kwargs):
        if getattr(self, 'recording_length_t', None) is None:
            raise NoDataError('The recording length must be provided. Units are to be in milliseconds.')

        return func(self, *args, **kwargs)

    return wrapper


def requires_spike_times(func):
    """ Decorator to ensure that 'spike_times' is provided in the instance. """

    def wrapper(self, *args, **kwargs):
        if getattr(self, 'spike_times', None) is None:
            raise NoDataError('The spike times must be provided.')

        return func(self, *args, **kwargs)

    return wrapper
