"""
ccg.py

Modules for plotting cross-correlograms.
"""

import pathlib

import numpy as np
import matplotlib.pyplot as plt

from synapticonn.utils.mod_utils import check_dependency
from synapticonn.plots.style import apply_plot_style
from synapticonn.plots.save import savefig
from synapticonn.plots.plot_utils import check_ax
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
@check_ax
def plot_ccg(cross_correlograms_data, labels=None, ax=None, **kwargs):
    """
    Plot cross-correlograms for spike train pairs with multiple subplots.

    Parameters
    ----------
    cross_correlograms_data : dict
        Dictionary containing 'cross_correlations' and 'bins'.
    bin_size_ms : float, optional
        Bin size of the cross-correlogram (in milliseconds). Default is 1 ms.
    max_lag_ms : float, optional
        Maximum lag to compute the cross-correlogram (in milliseconds). Default is 100 ms.
    labels : list of str or bool, optional
        List of labels for the spike trains or a flag to indicate whether to add axis labels. Default is True.
    ax : matplotlib.axes.Axes or np.ndarray of Axes, optional
        Axis or array of axes to plot on.
    num_axes : int, optional
        Number of axes (subplots) to create. Required if not providing an existing ax.
    ncols : int, optional
        Number of columns for the subplot grid. Required if num_axes is provided.
    figsize : tuple, optional
        Size of the figure.
    figtitle : str, optional
        Title of the figure.
    **kwargs
        Additional keyword arguments passed to ax.bar.

    Returns
    -------
    ax : numpy.ndarray of matplotlib.axes.Axes
        Array of axes with the cross-correlogram plots.
    """

    # handle labels
    if isinstance(labels, bool):
        if labels:
            labels = list(cross_correlograms_data['cross_correllations'].keys())
        else:
            labels = [''] * len(cross_correlograms_data['cross_correllations'])

    pair_identifiers = list(cross_correlograms_data['cross_correllations'].keys())

    # convert to str
    if not all(isinstance(label, str) for label in labels):
        labels = [str(label) for label in labels]
    if not all(isinstance(pair_id, str) for pair_id in pair_identifiers):
        pair_identifiers = [str(pair_id) for pair_id in pair_identifiers]

    # plot the cross-correlograms between all spike-unit pairs
    for pair_id in pair_identifiers:
        cross_corr = cross_correlograms_data['cross_correllations'][pair_id]
        bins = cross_correlograms_data['bins'][pair_id]
        bin_size = bins[1] - bins[0]

        # extract indices for the subplot grid
        label_x, label_y = pair_id.split('_')
        try:
            x = np.where(np.array(labels) == label_x)[0][0]
            y = np.where(np.array(labels) == label_y)[0][0]
        except IndexError:
            raise ValueError(f"Label {label_x} or {label_y} not found in spike train labels.")

        # plot on the subplot
        if x == y:
            ax[x, y].bar(bins[:-1], cross_corr, width=bin_size, align='edge', color=CCG_COLORS['self'], **kwargs)
        else:
            ax[x, y].bar(bins[:-1], cross_corr, width=bin_size, align='edge', color=CCG_COLORS['pairs'], **kwargs)

        # labeling
        if x == 0:
            ax[x, y].set_title(labels[y])
        if y == 0:
            ax[x, y].set_ylabel(labels[x])
        if x == len(labels) - 1:
            ax[x, y].set_xlabel('Time lag (ms)')

    return ax


# place the colors into a style folder??
# check how figsize is added