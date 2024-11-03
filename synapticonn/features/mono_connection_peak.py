""" mono_connection_peak.py 

Modules for computing the features of cross-correlogram (CCG) peaks.
"""

import numpy as np 


##################################################################
##################################################################


def compute_peak_amp(ccg):
    """ Compute the peak amplitude of the CCG.

    Parameters
    ----------
    ccg : array
        Cross-correlogram.

    Returns
    -------
    output : float
        Amplitude of the peak of the CCG
    """

    # find the peak of the cross-correlogram
    peak = np.max(ccg)

    return {'ccg_peak_amp': peak}