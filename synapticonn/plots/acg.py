""" acg.py

Modules for plotting autocorrelograms.
"""

import pathlib

import numpy as np
import matplotlib.pyplot as plt

from synapticonn.postprocessing.autocorrelograms import compute_autocorrelogram
from synapticonn.utils.mod_utils import check_dependency
from synapticonn.plots.spiketrain_utils import check_spiketrain_ndim, check_spiketrain_millisecond
from synapticonn.plots.plot_utils import acg_ax, check_spktime_ax_length
from synapticonn.plots.style import apply_plot_style
from synapticonn.plots.save import savefig


##########################################################
##########################################################


style_path = pathlib.Path('synapticonn', 'plots', 'settings.mplstyle')
plt.style.use(style_path)  # set globally


##########################################################
##########################################################


@savefig
@apply_plot_style(style_path=pathlib.Path('synapticonn', 'plots', 'settings.mplstyle'))
@check_dependency(plt, 'matplotlib')
@check_spktime_ax_length
@acg_ax
def plot_acg(spike_times, bin_size_t=1, max_lag_t=100, show_axes=True, ax=None, **kwargs):
    """Plot an autocorrelogram for a single spike train.

    Parameters
    ----------
    spike_times : dict
        Spike times (in milliseconds).
        Each key is a unit ID and each value is a list of spike times.
    bin_size_t : float
        Bin size of the autocorrelogram (in milliseconds).
        Default is 1 ms.
    max_lag_t : float
        Maximum lag to compute the autocorrelogram (in milliseconds).
        Default is 100 ms.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on.
    show_axes: bool, optional
        Whether to add axis labels. Default
    **kwargs
        Additional keyword arguments passed to `ax.bar`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axis with the autocorrelogram plot.
    """

    if not isinstance(spike_times, dict):
        msg = ("Spike train must be a dictionary. "
               "Each key is a unit ID. "
               "Each row is the corresponding spk times.")
        raise ValueError(msg)

    for count, (unit_id, single_spike_times) in enumerate(spike_times.items()):
        lags, autocorr = compute_autocorrelogram(single_spike_times, bin_size_t, max_lag_t)

        ax[count].bar(lags, autocorr, width=bin_size_t, align='center', **kwargs)
        ax[count].set_title(f'Unit {unit_id}')

        if show_axes:
            ax[count].set_xlabel('Time lag (ms)')
            ax[count].set_ylabel('Spike counts/bin')

    return ax
