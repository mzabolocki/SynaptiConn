""" ccg_connection_type.py

Modules to categorize connection types
based on synaptic strengths computed
using the cross-correlogram (CCG) method.
"""

import warnings
import numpy as np

from synapticonn.utils.errors import ConnectionTypeError


##########################################################
##########################################################


def get_putative_connection_type(synaptic_strength, threshold=None):
    """ Calculate putative connection type based on synaptic strengths.

    Parameters
    ----------
    synaptic_strength : float
        Synaptic strength value.
    threshold : float, optional
        Threshold value for categorizing connection types. Default is None.

    References
    ----------
    [1] STAR Protoc. 2024 Jun 21;5(2):103035. doi: 10.1016/j.xpro.2024.103035. Epub 2024 Apr 27
    """

    assert isinstance(synaptic_strength, (int, float, np.number)), "Synaptic strength must be a number."
    assert threshold is None or isinstance(threshold, (int, float, np.number)), "Threshold must be a number."

    if threshold > 0:
        return _get_exc_putative_connection_type(synaptic_strength, threshold)
    elif threshold < 0:
        return _get_inh_putative_connection_type(synaptic_strength, threshold)
    else:
        raise ConnectionTypeError("Threshold must be non-zero.")


def _get_exc_putative_connection_type(synaptic_strength, threshold=5):
    """ Categorize excitatory connection types based on synaptic strengths. """

    if threshold < 5:
        warnings.warn("Threshold is < 5. Recommended to use a threshold of >= 5 for excitatory connections.")

    if synaptic_strength >= threshold:
        label = "excitatory monosynaptic"
    elif synaptic_strength < threshold:
        label = 'undefined'

    return {'putative_exc_connection_type': label}


def _get_inh_putative_connection_type(synaptic_strength, threshold=5):
    """ Categorize inhibitory connection types based on synaptic strengths. """

    if threshold > -5:
        warnings.warn("Threshold is > -5. Recommended to use a threshold of <= -5 for inhibitory connections.")

    if synaptic_strength <= threshold:
        label = "inhibitory monosynaptic"
    elif synaptic_strength > threshold:
        label = 'undefined'

    return {'putative_inh_connection_type': label}