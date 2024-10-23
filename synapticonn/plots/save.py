""" save.py

Module for saving out figures.
"""

import matplotlib.pyplot as plt
from functools import wraps

from ..utils.file_io import fpath, fname


#############################################
#############################################


def savefig(func):
    """Decorator function to save out figures."""

    @wraps(func)
    def decorated(*args, **kwargs):

        # grab file name and path arguments, if they are in kwargs
        file_name = kwargs.pop('file_name', None)
        file_path = kwargs.pop('file_path', None)
        extension = kwargs.pop('extension', 'png')

        # check for an explicit argument for whether to save figure or not
        #   defaults to saving when file name given (since bool(str)->True; bool(None)->False)
        save_fig = kwargs.pop('save_fig', bool(file_name))

        # check any collect any other plot keywords
        save_kwargs = kwargs.pop('save_kwargs', {})
        save_kwargs.setdefault('bbox_inches', 'tight')

        # check and collect whether to close the plot
        close = kwargs.pop('close', None)

        func(*args, **kwargs)

        if save_fig:
            save_figure(file_name, file_path, close, extension, **save_kwargs)

    return decorated


def save_figure(file_name, file_path=None, close=False, extension='png', **save_kwargs):
    """Save out a figure.

    Parameters
    ----------
    file_name : str
        File name for the figure file to save out.
    file_path : Path or str
        Path for where to save out the figure to.
    close : bool, optional, default: False
        Whether to close the plot after saving.
    extension : str, optional, default: 'png'
        File extension for the figure file to save out.
    save_kwargs
        Additional arguments to pass into the save function.
    """

    full_path = fpath(file_path, fname(file_name, extension))
    plt.savefig(full_path, **save_kwargs)

    if close:
        plt.close()
