"""Style and aesthetics definitions for plots."""

import pathlib

from itertools import cycle
from functools import wraps

from matplotlib.pyplot import style

style.use(pathlib.Path('synapticonn', 'plots', 'plots.mplstyle'))