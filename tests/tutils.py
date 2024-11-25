""" tutils.py

Utilities for testing synapticonn."""

import pathlib
import numpy as np
import pandas as pd
import scipy.io

import synapticonn as synapticonn


###############################################################################
###############################################################################


def test_data_path(data_file_type: str) -> pathlib.Path:
    """ Return the path to the test data directory. """

    base_path = pathlib.Path(__file__).parent / "spiketimetest"

    if data_file_type == '.mat':
        # data available for the following reference: 
        # https://star-protocols.cell.com/protocols/3438
        # also found here on the github page:
        # https://github.com/matildebalbilab/STARProtocol_Wangetal2024 
        data_path = pathlib.Path(base_path, "mat_file", "all_unit.mat")
    elif data_file_type == 'spikeinterface':
        data_path = pathlib.Path(base_path, "spikeinterface", "BD0187_spikesorting_array.pkl")
    else:
        raise ValueError(f"Unknown data file type: {data_file_type}")

    return data_path


def load_mat_file(data_path):
    """ Load the .mat file and return the spiketimes. """

    data = scipy.io.loadmat(data_path)
    num_units = len(data['unit_t'][0])

    spiketimes = {}
    for i in range(num_units):
        spiketimes[i] = data['unit_t'][0][i].T[0]

    return spiketimes


def load_spikeinterface_file(data_path):
    """ Load the *.pkl spikeinterface file and return the spiketimes. """

    if data_path.suffix != '.pkl':
        raise ValueError(f"File is not a pickle file: {data_path}")

    data = np.load(data_path, allow_pickle=True)

    spiketimes = data['spike_time_set']

    # convert the spiketimes to milliseconds
    for i in spiketimes.keys():
        spiketimes[i] = (spiketimes[i] / 30000) * 1000

    return spiketimes


def load_spiketimes(data_file_type: str) -> pd.DataFrame:
    """ Load the spiketimes from the test data directory. """

    data_path = test_data_path(data_file_type)

    if data_file_type == '.mat':
        return load_mat_file(data_path)
    elif data_file_type == 'spikeinterface':
        return load_spikeinterface_file(data_path)
    else:
        raise ValueError(f"Unknown data file type: {data_file_type}")
