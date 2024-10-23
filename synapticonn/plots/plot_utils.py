""" plot_utils.py

Decorators for plotting utilities.

Notes
------
These should be considered private.
They are not expected to be used outside of this module or used
directly by the user.
"""

import numpy as np
import matplotlib.pyplot as plt


####################################################
####################################################


def check_ax(func):
    """ Decorator to check axes for spike-unit labels before plotting multiple subplots. """
    def wrapper(*args, **kwargs):

        ax = kwargs.get('ax', None)
        figsize = kwargs.pop('figsize', (25, 25))
        labels = kwargs.get('labels', None)

        # if labels are provided, make a grid of subplots
        if labels is not None:
            if ax is None:
                num_rows = int(np.ceil(len(labels)))
                num_cols = int(np.ceil(len(labels)))
                _, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
                kwargs['ax'] = ax
            else:
                # ensure the shape of ax matches the number of labels
                if (ax.shape[0] != len(labels)) or (ax.shape[1] != len(labels)):
                    raise ValueError("Number of axes must match the number of labels.")

        # if labels are not provided, return a single axis
        elif labels is None:
            if ax is None:
                ax = plt.gca()

        return func(*args, **kwargs)
    return wrapper
