""" connection_type.py

Modules to categorize connection types
based on synaptic strengths.
"""

import warnings
import numpy as np

from synapticonn.utils.errors import ConnectionTypeError


##########################################################
##########################################################


def get_putative_connection_type(synaptic_strength, threshold=5):
    """ Categorize excitatory connection types based on synaptic strengths.

    Parameters
    ----------
    synaptic_strength : float
        Synaptic strength value.
    threshold : float, optional
        Threshold value for excitatory connections. Default is 5.

    References
    ----------
    [1] STAR Protoc. 2024 Jun 21;5(2):103035. doi: 10.1016/j.xpro.2024.103035. Epub 2024 Apr 27
    """

    if threshold <= 0:
        raise ConnectionTypeError("Threshold must be > 0 for excitatory connections.")
    if threshold < 5:
        warnings.warn("Threshold is < 5. Recommended to use a threshold of >= 5 for excitatory connections.")

    return "excitatory monosynaptic" if synaptic_strength >= threshold else None


def get_inh_connection_type(synaptic_strength, threshold=5):
    """ Categorize inhibitory connection types based on synaptic strengths.

    Parameters
    ----------
    synaptic_strength : float
        Synaptic strength value.
    threshold : float, optional
        Threshold value for inhibitory connections. Default is -5.

    References
    ----------
    [1] STAR Protoc. 2024 Jun 21;5(2):103035. doi: 10.1016/j.xpro.2024.103035. Epub 2024 Apr 27
    """

    if threshold >= 0:
        raise ConnectionTypeError("Threshold must be < 0 for inhibitory connections.")
    if threshold > -5:
        warnings.warn("Threshold is > -5. Recommended to use a threshold of <= -5 for inhibitory connections.")

    return "inhibitory monosynaptic" if synaptic_strength <= threshold else None