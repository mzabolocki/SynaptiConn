"""Style and aesthetics definitions for plots."""

import pathlib

from itertools import cycle
from functools import wraps

from matplotlib.pyplot import style
import matplotlib.pyplot as plt


###################################################################
###################################################################


def apply_plot_style(style_path=None):
    """Decorator to apply matplotlib style before a plotting function."""

    if style_path is None:
        # Default style path
        style_path = pathlib.Path('synapticonn', 'plots', 'plots.mplstyle')

    def decorator_plot_style(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Apply the style before plotting
            plt.style.use(style_path)
            return func(*args, **kwargs)
        return wrapper

    return decorator_plot_style