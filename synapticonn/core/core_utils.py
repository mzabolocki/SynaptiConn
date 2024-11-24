""" core_utils.py

Utility functions for synapticonn package
with reusable utilities.
"""

import pathlib
import logging
import warnings

from functools import wraps
from typing import Any, List, Tuple

from synapticonn.utils.errors import SpikePairError, SpikeTimesError


######################################################
######################################################


## logging decorator

def setup_log(log_folder_name: str = 'removed_spike_units',
              log_fname: str = 'low_quality_units_removed.log'):
    """ Setup logging for specific class methods.

    Parameters
    ----------
    log_folder_name : str
        Name of the log folder to store the log file.
    log_fname : str
        Name of the log file.
    """

    log_folder = pathlib.Path('logs', log_folder_name)
    log_folder.mkdir(parents=True, exist_ok=True)
    log_path = pathlib.Path(log_folder, log_fname).absolute()
    logging.basicConfig(filename=log_path,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO,
                        force=True)


## helper check decorator


def _spike_pairs_check(self, spike_pairs):
    """ Check if the spike pairs are valid.

    Parameters
    ----------
    spike_pairs : List[Tuple]
        List of spike pairs to compute synaptic strength.

    Returns
    -------
    spike_pairs : List[Tuple]
        List of valid spike pairs.
    """

    if spike_pairs is None:
        raise SpikePairError("Please provide spike pairs to compute synaptic strength.")

    # check type
    if not isinstance(spike_pairs, List):
        raise SpikeTimesError("Spike pairs must be a list of tuples.")
    elif not all(isinstance(pair, Tuple) for pair in spike_pairs):
        raise SpikeTimesError("Spike pairs must be a list of tuples.")
    return spike_pairs


def _validate_parameter(self, name,
                        value,
                        min_value=None,
                        max_value=None,
                        warn_threshold=None,
                        warn_message=None):
    """ Generic validator for parameters with thresholds and warnings.

    Parameters
    ----------
    name : str
        Name of the parameter.
    value : float
        Value of the parameter.
    min_value : float
        Minimum value of the parameter.
    max_value : float
        Maximum value of the parameter.
    warn_threshold : float
        Warning threshold for the parameter.
    warn_message : str
        Warning message for the parameter.
    """

    if min_value is not None and value <= min_value:
        raise ValueError(f"{name} must be greater than {min_value}.")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} is greater than the allowed maximum ({max_value}). Adjust the value.")
    if warn_threshold is not None and value > warn_threshold:
        warnings.warn(warn_message, UserWarning)