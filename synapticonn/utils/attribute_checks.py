""" attribute_checks.py

Utils checking attributes of an object.
"""

from ..utils.errors import SamplingRateError, RecordingLengthError


#######################################################
#######################################################


def requires_sampling_rate(func):
    """ Decorator to ensure that 'srate' (sampling rate) is provided in the instance. """

    def wrapper(self, *args, **kwargs):
        if getattr(self, 'srate', None) is None:
            raise SamplingRateError('The sampling rate (srate) must be provided.')

        return func(self, *args, **kwargs)

    return wrapper


def requires_recording_length(func):
    """ Decorator to ensure that 'recording_length' is provided in the instance. """

    def wrapper(self, *args, **kwargs):
        if getattr(self, 'recording_length', None) is None:
            raise RecordingLengthError('The recording length must be provided.')

        return func(self, *args, **kwargs)

    return wrapper
