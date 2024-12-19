""" mono_connection_lateny.py 

Modules for computing the timing of cross-correlogram (CCG) peaks.
"""

import numpy as np 


##################################################################
##################################################################


def compute_peak_latency(ccg, bin_size_t):
    """ Compute the peak latency of the CCG.

    Parameters
    ----------
    ccg : array
        Cross-correlogram.
    bin_size_t : float
        Bin size of the cross-correlogram.
        This is the time difference between bins, 
        not there centres.

    Returns
    -------
    output : float
        Timing of the peak of the CCG
        
    Notes
    -----
    The bin size is the time difference between bins, not the bin width.
    Here, we calculate the time difference between the peak and the center of the CCG.
    """

    # find the peak of the cross-correlogram
    peak = np.argmax(ccg)

    # convert the peak to time
    peak_time = (peak - len(ccg) // 2) * (bin_size_t / 2) # time difference between bin centers

    return {'ccg_peak_time_ms': peak_time}
