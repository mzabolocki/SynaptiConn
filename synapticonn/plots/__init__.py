"""Sub-module for plot functions."""

from .acg import plot_acg
from .ccg import plot_ccg
from .synaptic_strength_calc import plot_synaptic_strength
from .spiketrain_utils import check_spiketrain_ndim, check_spiketrain_millisecond
from .plot_utils import check_ax, check_spktime_ax_length, check_ccg_ax
from .style import apply_plot_style