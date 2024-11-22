""" attribute_checks.py

Utils checking attributes of an object if present or not.
"""

import inspect
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
            raise NoDataError('The recording length must be provided.')

        return func(self, *args, **kwargs)

    return wrapper


def requires_spike_times(func):
    """ Decorator to ensure that 'spike_times' is provided in the instance. """

    def wrapper(self, *args, **kwargs):
        if getattr(self, 'spike_times', None) is None:
            raise NoDataError('The spike times must be provided.')

        return func(self, *args, **kwargs)

    return wrapper


def requires_arguments(*arg_names):
    """ Decorator to ensure specific arguments are provided. """

    def decorator(func):
        def wrapper(*args, **kwargs):

            # get the function's signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # check for all required arguments
            missing_args = [arg for arg in arg_names if bound_args.arguments.get(arg) is None]
            if missing_args:
                raise NoDataError(f"The following arguments are required but missing: {', '.join(missing_args)}")
            return func(*args, **kwargs)

        return wrapper
    return decorator