"""Utilities for testing synapticonn."""

import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import synapticonn as synapticonn

def test_data_path(data_file_type: str) -> pathlib.Path:
    """Return the path to the test data directory."""

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