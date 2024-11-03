""" ccg.py

Modules for plotting cross-correlograms.
"""

import pathlib

import numpy as np
import matplotlib.pyplot as plt

from synapticonn.utils.mod_utils import check_dependency
from synapticonn.plots.style import apply_plot_style
from synapticonn.plots.save import savefig
from synapticonn.plots.plot_utils import ccg_ax, check_spktime_ax_length
from synapticonn.plots.aesthetics import CCG_COLORS


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
@ccg_ax
def plot_ccg(cross_correlograms_data, ax=None, show_axes=True, **kwargs):
    """
    Plot cross-correlograms for spike train pairs with multiple subplots.

    Parameters
    ----------
    cross_correlograms_data : dict
        Dictionary containing 'cross_correlations' and 'bins' values.
        Can be outputted from `compute_crosscorrelogram` function in `crosscorrelograms.py`.
    ax : numpy.ndarray of matplotlib.axes.Axes
        Array of axes to plot on.
    show_axes : bool, optional
        Whether to add axis labels. Default is True.
    **kwargs
        Additional keyword arguments passed to ax.bar.

    Returns
    -------
    ax : numpy.ndarray of matplotlib.axes.Axes
        Array of axes with the cross-correlogram plots.
    """

    pair_identifiers = (cross_correlograms_data['cross_correllations'].keys())

    # plot the cross-correlograms between all spike-unit pairs
    for count, pair_id in enumerate(pair_identifiers):
        cross_corr = cross_correlograms_data['cross_correllations'][pair_id]
        bins = cross_correlograms_data['bins'][pair_id]
        bin_size = bins[1] - bins[0]

        ax[count].bar(bins[:-1], cross_corr, width=bin_size, align='edge', color=CCG_COLORS['pairs'], **kwargs)
        ax[count].set_title(f'Pair {pair_id}')

        if show_axes:
            ax[count].set_xlabel('Lag (ms)')
            ax[count].set_ylabel('Cross-correlation')

    return ax
