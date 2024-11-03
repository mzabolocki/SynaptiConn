""" mono_connection_lateny.py 

Modules for computing the timing of cross-correlogram (CCG) peaks.
"""

import numpy as np 


##################################################################
##################################################################


def compute_peak_latency(ccg, bin_size):
    """ Compute the peak latency of the CCG.

    Parameters
    ----------
    ccg : array
        Cross-correlogram.
    bin_size : float
        Bin size of the cross-correlogram.

    Returns
    -------
    output : float
        Timing of the peak of the CCG
    """

    # find the peak of the cross-correlogram
    peak = np.argmax(ccg)

    # convert the peak to time
    peak_time = (peak - len(ccg) // 2) * bin_size

    return {'ccg_peak_time_ms': peak_time}