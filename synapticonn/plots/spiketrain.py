""" spiketrain.py

Decorators for checking spiketrains before function execution.

Notes
------
These should be considered private.
They are not expected to be used outside of this module or used
directly by the user.
"""

import pathlib
import pkg_resources
import numpy as np
import matplotlib.pyplot as plt

from synapticonn.utils.mod_utils import check_dependency
from synapticonn.plots.style import apply_plot_style
from synapticonn.plots.save import savefig
from synapticonn.plots.plot_utils import spktrain_ax
from synapticonn.plots.aesthetics import TIME_UNIT_TIMELAG_LABELS


##########################################################
##########################################################


style_path = pkg_resources.resource_filename('synapticonn.plots', 'settings.mplstyle')
plt.style.use(style_path)


####################################################
####################################################


@savefig
@apply_plot_style(style_path=style_path)
@check_dependency(plt, 'matplotlib')
@spktrain_ax
def plot_spiketrain(spike_times,
                    ax=None,
                    time_unit='ms',
                    **kwargs):
    """ Plot a spike train.

    Parameters
    ----------
    spike_times : dict
        Spike times.
        Each key is a unit ID and each value is a list of spike times.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on.
    time_unit : str, optional
        Time unit for the x-axis. Default is 'ms'.
    **kwargs
        Additional keyword arguments passed to `ax.eventplot`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axis with the spike train plot.
    """

    for count, (spk_id, spk_times) in enumerate(spike_times.items()):
        ax.eventplot(spk_times, lineoffsets=count, **kwargs)

    # rename y-axis labels
    ax.set_yticks(np.arange(len(spike_times)))
    ax.set_yticklabels(spike_times.keys())

    # labels
    ax.set_ylabel('Unit ID')
    ax.set_xlabel(TIME_UNIT_TIMELAG_LABELS[time_unit])

    return ax


def check_spiketrain_ndim(func):
    """Decorator to check if array is 1D before function execution.
    To be used on a single spike train (1D), in a list or array format.
    """

    def wrapper(spike_train_ms, *args, **kwargs):
        if len(spike_train_ms) == 0:
            raise ValueError("Array is empty.")
        if spike_train_ms.ndim != 1:
            raise ValueError("Array must be 1D.")
        return func(spike_train_ms, *args, **kwargs)

    return wrapper


def check_spiketrain_millisecond(func):
    """ Decorator to check if array is in milliseconds before function execution.

    To be performed on a single spike train (1D).

    Notes:
    ------
    This assumes that spike times are not in milliseconds if the minimum ISI
    is less than 0,1. Spike ISI from individual neurons should not fire
    faster than 1 ms, so this is a reasonable assumption.
    """

    def wrapper(spike_train_ms, *args, **kwargs):

        min_isi_thresh = 0.1
        msg = ("Check spike times. "
               "If values are not in milliseconds, convert to milliseconds. "
               "Minimum ISI is < 0.1.")

        # flatten spike_train_ms in case it's multi-dimensional
        # handle both 1D and 2D spike trains uniformly
        spike_train_ms = np.atleast_1d(spike_train_ms)

        if spike_train_ms.size == 0:
            raise ValueError("Spike train is empty. Cannot check ISI.")

        # calculate ISI (inter-spike intervals) for the entire array
        isis = np.diff(spike_train_ms, axis=-1)

        # find the minimum ISI across all spike trains
        min_isi = np.min(isis)

        # raise an error if the minimum ISI is below the threshold
        if min_isi < min_isi_thresh:
            raise ValueError(f"{msg} Minimum ISI found: {min_isi} ms.")

        # continue to the wrapped function if no issue
        return func(spike_train_ms, *args, **kwargs)

    return wrapper
