""" report.py

Utilities for generating reports.

Note that this is a modified version of the report.py file from the neurodsp package
and fooof package, with the original source code available at:
https://github.com/fooof-tools/fooof/blob/main/specparam/core/strings.py#L266
"""

import numpy as np
import pandas as pd


###################################################################################################
###################################################################################################

## Settings & Globals
# Centering Value - Long & Short options
#   Note: Long CV of 98 is so that the max line length plays nice with notebook rendering
LCV = 98
SCV = 70

###################################################################################################
###################################################################################################


def gen_model_results_str(connection_types, concise, params):
    """Generate a string representation of model monosynaptic connection inference results.

    Parameters
    ----------
    connection_types : dict
        Dictionary of connection types, and the corresponding model results.
    concise : bool, optional, default: False
        Whether to print the report in concise mode.
    params : key-value pairs
        Additional parameters to include in the report
        used for computing the model.

    Returns
    -------
    output : str
        Formatted string of monosynaptic connection results.
    """

    if params.get('method') == 'ccg':

        # params
        ccg_binsize_ms = params.get('bin_size_ms')
        max_lag_ms = params.get('max_lag_ms')
        srate = params.get('srate')
        recording_length_ms = params.get('recording_length_ms')
        recording_length_sec = recording_length_ms / 1000
        synaptic_strength_threshold = params.get('synaptic_strength_threshold')
        bootstrap_n = params.get('n_bootstraps')

        # summarise the results
        connections = pd.DataFrame(connection_types).T
        exc_count = sum(connections.putative_exc_connection_type == 'excitatory')
        inh_count = sum(connections.putative_exc_connection_type == 'inhibitory')

    else:
        raise ValueError('Method not recognized.')

    # create the formatted strings for printing
    str_lst = [

        # header
        '=',
        '',
        'SYNAPTICONN - MONOSYNAPTIC CONNECTIONS',
        '',

        # ccg method parameters
        'CCG Method Parameters:',
        '',
        'Sampling frequency is {:1.2f} Hz'.format(srate),
        'Recording length is {:1.2f} seconds'.format(recording_length_sec),
        'CCG bin size is {:1.2f} ms'.format(ccg_binsize_ms),
        'Maximum lag is {:1.2f} ms'.format(max_lag_ms),
        'Synaptic strength threshold cut-off is {:1.2f}'.format(synaptic_strength_threshold),
        'Number of iterations for jitter: {}'.format(bootstrap_n),
        '',

        # connection types
        'Connection Types:',
        '',
        'Number of excitatory connections: {}'.format(exc_count),
        'Number of inhibitory connections: {}'.format(inh_count),
        '',

        # footer
        '='
    ]

    output = _format(str_lst, concise)

    return output


def _format(str_lst, concise):
    """Format a string for printing.

    Parameters
    ----------
    str_lst : list of str
        List containing all elements for the string, each element representing a line.
    concise : bool, optional, default: False
        Whether to print the report in a concise mode, or not.

    Returns
    -------
    output : str
        Formatted string, ready for printing.
    """

    # Set centering value - use a smaller value if in concise mode
    center_val = SCV if concise else LCV

    # Expand the section markers to full width
    str_lst[0] = str_lst[0] * center_val
    str_lst[-1] = str_lst[-1] * center_val

    # Drop blank lines, if concise
    str_lst = list(filter(lambda x: x != '', str_lst)) if concise else str_lst

    # Convert list to a single string representation, centering each line
    output = '\n'.join([string.center(center_val) for string in str_lst])

    return output
