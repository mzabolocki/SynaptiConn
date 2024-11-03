""" ccg.py 

Modules for computing cross-correlogram (CCG) features.
"""

import numpy as np 


##################################################################
##################################################################


def compute_ccg_bootstrap(ccg, n_bootstraps=1000):
    """ Compute the bootstrap confidence interval of the CCG.

    Parameters
    ----------
    ccg : array
        Cross-correlogram.
    n_bootstraps : int
        Number of bootstrap samples to draw.

    Returns
    -------
    output : dict
        Dictionary containing the bootstrap standard deviation and the number of bootstraps.
    """

    # arr to store bootstrap samples' means for each iteration
    bootstrap_means = np.zeros(n_bootstraps)

    # perform bootstrap resampling
    for i in range(n_bootstraps):
        bootstrap_sample = np.random.choice(ccg, size=len(ccg), replace=True)
        bootstrap_means[i] = np.mean(bootstrap_sample)

    # calculate confidence intervals (e.g., 95%)
    bootstrap_std = np.std(bootstrap_means)

    return {'bootstrap_std': bootstrap_std, 'n_bootstraps': n_bootstraps}


def compute_ccg_cv(ccg):
    """Compute the coefficient of variation of the CCG.

    Parameters
    ----------
    ccg : array
        Cross-correlogram.

    Returns
    -------
    output : dict
        Dictionary containing the coefficient of variation of the CCG.
    """

    # compute the mean and standard deviation of the CCG
    ccg_mean = np.mean(ccg)
    ccg_std = np.std(ccg)

    # compute the coefficient of variation of the CCG
    ccg_cv = ccg_std / ccg_mean

    return {'ccg_cv': ccg_cv}

