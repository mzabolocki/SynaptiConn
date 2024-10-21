""" checks.py

Decorators for checking array properties before function execution.
"""


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
