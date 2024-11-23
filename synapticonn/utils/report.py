""" report.py

Utilities for generating reports.

Note that this is a modified version of the report.py file from the neurodsp package
and fooof package, with the original source code available at:
https://github.com/fooof-tools/fooof/blob/main/specparam/core/strings.py#L266
"""

import pandas as pd


###################################################################################################
###################################################################################################

## Settings & Globals
# Centering Value - Long & Short options
# Note: Long CV of 98 is so that the max line length plays nice with notebook rendering
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

    if params.get('method') == 'cross-correlation':

        # params
        ccg_binsize_ms = params.get('bin_size_t')
        max_lag_t = params.get('max_lag_t')
        srate = params.get('srate')
        recording_length_t = params.get('recording_length_t')
        recording_length_sec = recording_length_t / 1000
        synaptic_strength_threshold = params.get('synaptic_strength_threshold')
        num_iterations = params.get('num_iterations')
        half_window_t = params.get('half_window_t')
        jitter_range_t = params.get('jitter_range_t')

        # count the number of excitatory and inhibitory connections
        connections = pd.DataFrame(connection_types).T
        exc_connections = connections[connections.putative_exc_connection_type == 'excitatory']
        inh_connections = connections[connections.putative_exc_connection_type == 'inhibitory']

        exc_count = len(exc_connections)
        inh_count = len(inh_connections)

        # identify the pairs
        exc_pairs = exc_connections.index.values
        inh_pairs = inh_connections.index.values

    else:
        raise ValueError('Method not recognized.')

    # create the formatted strings for printing
    str_lst = [

        # header
        '=',
        '',
        'SYNAPTICONN - MONOSYNAPTIC CONNECTIONS',
        '',
        '----------------------------------------',
        # recording parameters
        'Recording Parameters:',
        '',
        'Sampling rate: {} Hz'.format(srate),
        'Recording length: {:1.2f} s'.format(recording_length_sec),
        '',

        # ccg method parameters
        'Cross-Correlation Method Parameters:',
        '',
        'Crosscorrelogram bin size: {:1.2f} ms'.format(ccg_binsize_ms),
        'Maximum time lag: {:1.2f} ms'.format(max_lag_t),
        'Synaptic strength threshold cut-off: {:1.2f}'.format(synaptic_strength_threshold),
        'Half window size used to calculate synaptic strength: {} ms'.format(half_window_t),
        'Jitter range for synaptic strength computation: {:1.2f} ms'.format(jitter_range_t),
        'Number of iterations for jitter: {}'.format(num_iterations),
        ''
        '',
        '----------------------------------------',
        '',
    ]

    # add excitatory connection types if exc_pairs exists
    if exc_pairs:
        str_lst.append(f'{exc_count} excitatory connections were found:')
        str_lst.extend(map(str, exc_pairs))
    else:
        str_lst.append('No excitatory connection types were found.')

    str_lst.append('')

    # add inhibitory connections
    if inh_pairs:
        str_lst.append(f'{inh_count} inhibitory connections were found:')
        str_lst.extend(map(str, inh_pairs))
    else:
        str_lst.append('No inhibitory connection types were found.')

    str_lst.append('')

    # footer
    str_lst.extend([
        '',
        '='
    ])

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
