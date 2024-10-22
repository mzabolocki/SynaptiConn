""" checks.py

Decorators for checking array properties before function execution.
"""

import numpy as np


####################################################
####################################################


def check_ndim(func):
    """Decorator to check if array is 1D before function execution."""

    def wrapper(spike_train_ms, *args, **kwargs):
        if spike_train_ms.ndim != 1:
            raise ValueError("Array must be 1D.")
        return func(spike_train_ms, *args, **kwargs)

    return wrapper


def check_empty(func):
    """Decorator to check if array is not empty before function execution."""

    def wrapper(spike_train_ms, *args, **kwargs):
        if len(spike_train_ms) == 0:
            raise ValueError("Array is empty.")
        return func(spike_train_ms, *args, **kwargs)

    return wrapper


def check_millisecond(func):
    """ Decorator to check if array is in milliseconds before function execution.

    Notes:
    ------
    This assumes that spike times are not in milliseconds if the minimum ISI
    is less than 0,1. Spike ISI from individual neurons should not fire
    faster than 1 ms, so this is a reasonable assumption.
    """

    def wrapper(spike_train_ms, *args, **kwargs):

        # infer time unit based on isis and spike time range
        isi = np.diff(spike_train_ms)
        min_isi = np.min(isi)

        if min_isi < 0.1:
            msg = ("Check spike times. "
                   "If values are not in milliseconds, convert to milliseconds. "
                   "Minimum ISI is < 0.1.")
            raise ValueError(msg)

        else:
            pass  # assume values are in milliseconds

        return func(spike_train_ms, *args, **kwargs)

    return wrapper
