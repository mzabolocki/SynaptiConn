""" plot_utils.py

Decorators for plotting utilities.

Notes
------
These should be considered private.
They are not expected to be used outside of this module or used
directly by the user.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

from synapticonn.utils.errors import PlottingError


####################################################
####################################################


def acg_ax(func):
    """ Decorator to check axes for spike-unit labels before plotting multiple subplots.

    Note: this decorator is used for plotting multiple acg subplots. However, 
    to ccg plots, this decorator is not used.
    """
    def wrapper(spike_times, *args, **kwargs):

        n_units = len(spike_times)
        n_cols = min(n_units, 5)  # limit to 5 columns
        n_rows = math.ceil(n_units / n_cols)

        ax = kwargs.get('ax', None)
        figsize = kwargs.get('figsize', (15, 5 * n_rows))

        if ax is None:
            fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
            ax = ax.flatten() if isinstance(ax, np.ndarray) else [ax]
        elif ax is not None:
            pass

        kwargs['ax'] = ax
        kwargs.pop('figsize', None)

        return func(spike_times, *args, **kwargs)

    return wrapper


def ccg_ax(func):
    """ Decorator to check axes for spike-unit labels before plotting multiple subplots."""
    def wrapper(cross_correlograms_data, **kwargs):

        ax = kwargs.get('ax', None)
        figsize = kwargs.pop('figsize', (25, 25))

        if ax is None:
            n_pairs = len(list(cross_correlograms_data['cross_correllations'].keys()))
            n_cols = min(n_pairs, 5)  # limit to 5 rows
            n_rows = math.ceil(n_pairs / n_cols)

            _, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
            kwargs['ax'] = ax.flatten() if isinstance(ax, np.ndarray) else [ax]

        return func(cross_correlograms_data, **kwargs)
    return wrapper


def check_spktime_ax_length(func):
    """ Decorator to check axes length before plotting multiple subplots. """
    def wrapper(spike_times, *args, **kwargs):

        ax = kwargs.get('ax', None)

        if ax is not None:
            if not isinstance(ax, np.ndarray):  # single axes case
                ax = [ax]
            if len(ax) < len(spike_times):
                msg = ("Number of axes must be equal to the number of units.")
                raise PlottingError(msg)

        kwargs['ax'] = ax

        return func(spike_times, *args, **kwargs)

    return wrapper
