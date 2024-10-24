"""
mod_utils.py

Module utilities functions.
"""

from functools import wraps


#################################################
#################################################


def check_dependency(dep, name):
    """Decorator that checks if an optional dependency is available.

    Parameters
    ----------
    dep : module or False
        Module, if successfully imported, or boolean (False) if not.
    name : str
        Full name of the module, to be printed in message.

    Returns
    -------
    wrap : callable
        The decorated function.

    Raises
    ------
    ImportError
        If the requested dependency is not available.
    """

    def wrap(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if not dep:
                raise ImportError("Optional dependency " + name + \
                                  " is required for this functionality.")
            return func(*args, **kwargs)
        return wrapped_func
    return wrap


DOCSTRING_SECTIONS = ['Parameters', 'Returns', 'Yields', 'Raises',
                      'Warns', 'Examples', 'References', 'Notes',
                      'Attributes', 'Methods']
