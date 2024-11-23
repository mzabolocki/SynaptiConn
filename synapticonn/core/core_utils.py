""" core_utils.py

Utility functions for synapticonn package.
"""

import pathlib
import logging
from functools import wraps


######################################################
######################################################


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
