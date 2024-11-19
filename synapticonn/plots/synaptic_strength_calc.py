""" synaptic_strength_calc.py

Modules for plotting ccg and synaptic strength calculations.
"""

import pathlib

import numpy as np
import matplotlib.pyplot as plt

from synapticonn.utils.mod_utils import check_dependency
from synapticonn.plots.style import apply_plot_style
from synapticonn.plots.save import savefig
from synapticonn.plots.aesthetics import SYN_STRENGTH_COLORS


##########################################################
##########################################################


style_path = pathlib.Path('synapticonn', 'plots', 'settings.mplstyle')
plt.style.use(style_path)  # set globally


##########################################################
##########################################################


@savefig
@apply_plot_style(style_path=pathlib.Path('synapticonn', 'plots', 'settings.mplstyle'))
@check_dependency(plt, 'matplotlib')
def plot_ccg_synaptic_strength(pair_synaptic_strength_data, spike_pair, ax=None, bbox_to_anchor=(2.2, 1), **kwargs):
    """ Plots synaptic strength based on the provided data using the ccg method.

    Parameters
    ----------
    pair_synaptic_strength_data : dict
        A dictionary containing keys 'ccg_bins', 'original_ccg_counts',
        'jittered_ccg_counts', 'high_ci', 'low_ci', and 'window_slice'.
    spike_pair : tuple
        Spike pair for which synaptic strength is calculated.
    ax : list of matplotlib.axes._subplots.AxesSubplot, optional
        A list of two matplotlib axis objects. If None, new axes will be created.
    **kwargs : dict
        Additional keyword arguments for customizing the plot.

    Returns
    -------
    ax : list of matplotlib.axes._subplots.AxesSubplot
        A list of two matplotlib axis objects.
    """

    # unpack the data
    data = pair_synaptic_strength_data[spike_pair]
    ccg_bins, ccg_data = data['ccg_bins'], data['original_ccg_counts']
    jittered_ccg_counts = data['jittered_ccg_counts']
    high_ci, low_ci, window_slice = data['high_ci'], data['low_ci'], data['window_slice']

    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)
        fig.suptitle(f'Pair: {spike_pair}', fontsize=16)

    # plot the original and jittered ccgs
    ax[0].bar(ccg_bins[1:], ccg_data, width=0.5, color=SYN_STRENGTH_COLORS['original'])
    ax[1].bar(ccg_bins[1:], jittered_ccg_counts.mean(axis=0), width=0.5, color=SYN_STRENGTH_COLORS['jittered'])
    ax[0].plot(ccg_bins[1:], high_ci, color=SYN_STRENGTH_COLORS['ci'], alpha=1, ls='--', label='99% CI')
    ax[0].plot(ccg_bins[1:], low_ci, color=SYN_STRENGTH_COLORS['ci'], alpha=0.3, ls='--', label='1% CI')

    # highlight the selected window used for synaptic strength calculation
    for i in range(2):
        ymin, ymax = ax[i].get_ylim()
        rect = plt.Rectangle((ccg_bins[window_slice.start], 0),
                             ccg_bins[window_slice.stop] - ccg_bins[window_slice.start],
                             ymax, color=SYN_STRENGTH_COLORS['rectangle'], alpha=0.5, label='Selected window')
        ax[i].add_patch(rect)

    # labels
    ax[0].set_title('Original CCG')
    ax[1].set_title('Jittered CCG')

    for i in range(2):
        ax[i].set_xlabel('Time lag (ms)')
        ax[i].set_ylabel('Spike count/bin')

    ax[0].legend(bbox_to_anchor=bbox_to_anchor, loc='upper left')

    return ax
