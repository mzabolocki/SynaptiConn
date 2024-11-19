"""Sub-module for plot functions."""

from .acg import plot_acg
from .ccg import plot_ccg
from .synaptic_strength_calc import plot_ccg_synaptic_strength
from .spiketrain_utils import check_spiketrain_ndim, check_spiketrain_millisecond
from .plot_utils import acg_ax, check_spktime_ax_length, ccg_ax
from .style import apply_plot_style