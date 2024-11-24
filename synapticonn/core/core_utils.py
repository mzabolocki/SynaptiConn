""" core_utils.py

Utility functions for synapticonn package
with reusable utilities.
"""

import pathlib
import logging
import warnings

from functools import wraps
from typing import Any, List, Tuple

import numpy as np

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


def _validate_parameter(name,
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


def _validate_spike_pairs(spike_pairs: List[Tuple] = None,
                          spike_unit_ids: list = None):
    """ Validate spike pairs.

    Check the validity of spike pairs and filter them
    against the available spike unit IDs. These are
    found in the spike unit labels.

    Parameters
    ----------
    spike_pairs : List[Tuple]
        List of spike pairs.
    spike_unit_ids : list
        List of spike unit labels.
        This corresponds with the spike unit IDs
        in the spiketimes data.

    Returns
    -------
    valid_spike_pairs : List[Tuple]
        List of valid spike pairs.
    """

    # check the spike pairs type
    if spike_pairs is not None:
        if not isinstance(spike_pairs, list) or not \
                all(isinstance(pair, tuple) for pair in spike_pairs):
            raise SpikePairError("Spike pairs must be a list of tuples.")
    else:
        raise SpikePairError("Please provide spike pairs to "
                             "compute synaptic strength.")

    # check and convert the spike pairs to numpy array
        # this is for easier manipulation and filtering
    if not isinstance(spike_pairs, np.ndarray):
        spike_pairs = np.array(spike_pairs)

    # find the valid spike pairs
    valid_spike_units = spike_pairs[np.isin(spike_pairs, spike_unit_ids)]

    # raise error if no valid spike units
    if len(valid_spike_units) == 0:
        raise SpikeTimesError('No valid spike units to plot.')

    # isolate the valid and invalid spike pairs
    invalid_spike_pairs = [pair for pair in spike_pairs if pair[0] not in valid_spike_units \
                            or pair[1] not in valid_spike_units]
    valid_spike_pairs = [pair for pair in spike_pairs if pair[0] in valid_spike_units \
                            and pair[1] in valid_spike_units]

    # warn if invalid spike pairs are found
    if invalid_spike_pairs:
        warnings.warn(
            f"Invalid spike pairs found: {invalid_spike_pairs}. These pairs will be ignored.",
            UserWarning
        )

    return valid_spike_pairs

