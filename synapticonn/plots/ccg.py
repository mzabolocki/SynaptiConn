""" ccg.py

Modules for plotting cross-correlograms.
"""

import pathlib

import numpy as np
import matplotlib.pyplot as plt

from synapticonn.utils.mod_utils import check_dependency
from synapticonn.plots.style import apply_plot_style
from synapticonn.plots.save import savefig
from synapticonn.plots.plot_utils import check_ccg_ax, check_ax_length
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
# @check_ax_length
@check_ccg_ax
def plot_ccg(cross_correlograms_data, ax=None, **kwargs):
    """
    Plot cross-correlograms for spike train pairs with multiple subplots.

    Parameters
    ----------
    cross_correlograms_data : dict
        Dictionary containing 'cross_correlations' and 'bins' values.
        Can be outputted from `compute_crosscorrelogram` function in `crosscorrelograms.py`.
    **kwargs
        Additional keyword arguments passed to ax.bar.

    Returns
    -------
    ax : numpy.ndarray of matplotlib.axes.Axes
        Array of axes with the cross-correlogram plots.
    """

    # handle labels
    pair_identifiers = list(cross_correlograms_data['cross_correllations'].keys())
    pair_identifiers = [str(pair_id) for pair_id in pair_identifiers]
    spike_unit_labels = np.unique([pair_id.split('_') for pair_id in pair_identifiers])

    # plot the cross-correlograms between all spike-unit pairs
    for count, pair_id in enumerate(pair_identifiers):
        cross_corr = cross_correlograms_data['cross_correllations'][pair_id]
        bins = cross_correlograms_data['bins'][pair_id]
        bin_size = bins[1] - bins[0]

        # # extract indices for the subplot grid
        # label_x, label_y = pair_id.split('_')
        # try:
        #     x = np.where(np.array(spike_unit_labels) == label_x)[0][0]
        #     y = np.where(np.array(spike_unit_labels) == label_y)[0][0]
        # except IndexError:
        #     raise ValueError(f"Label {label_x} or {label_y} not found in spike train labels.")

        # # plot on the subplot
        # if x == y:
        #     ax[x, y].bar(bins[:-1], cross_corr, width=bin_size, align='edge', color=CCG_COLORS['self'], **kwargs)
        # else:
        #     ax[x, y].bar(bins[:-1], cross_corr, width=bin_size, align='edge', color=CCG_COLORS['pairs'], **kwargs)

        ax[count].bar(bins[:-1], cross_corr, width=bin_size, align='edge', color=CCG_COLORS['pairs'], **kwargs)

        # # labeling
        # if x == 0:
        #     ax[x, y].set_title(spike_unit_labels[y])
        # if y == 0:
        #     ax[x, y].set_ylabel(spike_unit_labels[x])
        # if x == len(spike_unit_labels) - 1:
        #     ax[x, y].set_xlabel('Time lag (ms)')

    return ax
