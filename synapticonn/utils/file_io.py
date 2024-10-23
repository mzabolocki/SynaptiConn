""" file_io.py

File input/output utilities.
"""

import os


###############################################################################
###############################################################################


def fname(file_name, extension):
    """Check a filename, adding an extension if not already specified.

    Parameters
    ----------
    file_name : str
        String that specifies a file name.
    extension : str
        String of the extension (without a period) to be added if one isn't already present.

    Returns
    -------
    file_name : str
        String that specifies a file name.
    """

    if len(file_name.split('.')) == 1:
        file_name = file_name + '.' + extension

    return file_name


def fpath(file_path, file_name):
    """Build the full file path from file name and directory.

    Parameters
    ----------
    file_path : Path or str or None
        Path to the directory where the file is located.
    file_name : str
        Name of the file.

    Returns
    -------
    full_path : str
        Full file path to the file, including directory, if provided.

    Notes
    -----
    This function is mainly used to deal with the case in which file_path is None.
    """

    if not file_path:
        full_path = file_name
    else:
        full_path = os.path.join(file_path, file_name)

    return full_path
