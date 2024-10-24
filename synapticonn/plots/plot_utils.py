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


####################################################
####################################################


def check_ax(func):
    """ Decorator to check axes for spike-unit labels before plotting multiple subplots. """
    def wrapper(spike_train_ms, *args, **kwargs):

        ax = kwargs.get('ax', None)
        n_units = len(spike_train_ms)

        # determine the number of rows and columns for the subplots
        if ax is None:
            n_cols = min(n_units, 5)  # limit to 5 columns
            n_rows = math.ceil(n_units / n_cols)

            fig, ax = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            ax = ax.flatten() if isinstance(ax, np.ndarray) else [ax]

            kwargs['ax'] = ax

        return func(spike_train_ms, *args, **kwargs)

    return wrapper
