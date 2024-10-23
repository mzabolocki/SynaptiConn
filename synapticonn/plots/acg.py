"""
acg.py

Modules for plotting autocorrelograms.
"""

import pathlib

import numpy as np
import matplotlib.pyplot as plt

from synapticonn.postprocessing.autocorrelograms import compute_autocorrelogram
from synapticonn.utils.mod_utils import check_dependency
from synapticonn.plots.spiketrain_utils import check_spiketrain_ndim, check_spiketrain_millisecond
from synapticonn.plots.plot_utils import check_ax
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
@check_spiketrain_ndim
@check_spiketrain_millisecond
@check_ax
def plot_acg(spike_train_ms, bin_size_ms=1, max_lag_ms=100, show_axes=True, ax=None, **kwargs):
    """Plot an autocorrelogram for a single spike train.

    Parameters
    ----------
    spike_train_ms : array-like
        Spike times (in milliseconds).
    bin_size_ms : float
        Bin size of the autocorrelogram (in milliseconds).
        Default is 1 ms.
    max_lag_ms : float
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

    lags, autocorr = compute_autocorrelogram(spike_train_ms, bin_size_ms, max_lag_ms)

    ax.bar(lags, autocorr, width=bin_size_ms, **kwargs)

    if show_axes:
        ax.set_xlabel('Time lag (ms)')
        ax.set_ylabel('Spike counts/bin')

    return ax


