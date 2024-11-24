""" data_utils.py 

Module for data utilities with reusable functions.
"""

import numpy as np


######################################################
######################################################


def flatten_list(vals):
    """ Flatten a 2D list to a 1D list.

    Parameters
    ----------
    arr : list
        2D list to be flattened.

    Returns
    -------
    Flattened 1D list.
    """

    if isinstance(vals, list):

        flattened_list = [int(item) for array in vals for item in array]
        unique_list = list(set(flattened_list))

        return unique_list