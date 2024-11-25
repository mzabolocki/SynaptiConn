""" acg.py

Modules for plotting autocorrelograms.
"""

import pathlib
import pkg_resources
import matplotlib.pyplot as plt

from synapticonn.postprocessing.autocorrelograms import compute_autocorrelogram
from synapticonn.utils.mod_utils import check_dependency
from synapticonn.plots.plot_utils import acg_ax, check_spktime_ax_length
from synapticonn.plots.style import apply_plot_style
from synapticonn.plots.save import savefig
from synapticonn.plots.aesthetics import TIME_UNIT_TIMELAG_LABELS


##########################################################
##########################################################


style_path = pkg_resources.resource_filename('synapticonn.plots', 'settings.mplstyle')
plt.style.use(style_path)


##########################################################
##########################################################


@savefig
@apply_plot_style(style_path=style_path)
@check_dependency(plt, 'matplotlib')
@check_spktime_ax_length
@acg_ax
def plot_acg(spike_times,
             bin_size_t=1,
             max_lag_t=100,
             show_axes=True,
             ax=None,
             time_unit='ms',
             **kwargs):
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
        is True.
    time_unit : str, optional
        The time unit for the x-axis. Default is 'ms'.
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
            ax[count].set_xlabel(TIME_UNIT_TIMELAG_LABELS[time_unit])
            ax[count].set_ylabel('Spike counts/bin')

    return ax
