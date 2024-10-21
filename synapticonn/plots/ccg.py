"""
ccg.py

Modules for plotting cross-correlograms.
"""

import pathlib

import numpy as np
import matplotlib.pyplot as plt

from synapticonn.utils.mod_utils import check_dependency
from synapticonn.plots.checks import check_empty, check_ndim
from synapticonn.plots.style import apply_plot_style


##########################################################
##########################################################


style_path = pathlib.Path('synapticonn', 'plots', 'plots.mplstyle')
plt.style.use(style_path)  # set globally


##########################################################
##########################################################


@apply_plot_style(style_path=pathlib.Path('synapticonn', 'plots', 'plots.mplstyle'))
@check_dependency(plt, 'matplotlib')
@check_ndim
@check_empty
def plot_ccg(spike_train_ms_1, spike_train_ms_2, bin_size_ms=1, max_lag_ms=100, labels=True, ax=None, **kwargs):
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
    labels: bool, optional
        Whether to add axis labels. Default
    **kwargs
        Additional keyword arguments passed to `ax.bar`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axis with the autocorrelogram plot.
    """

    if ax is None:
        ax = plt.gca()

