""" test_load.py

Test the spike time loader functions and checks in synapticonn.

NOTES
-----
The tests here are not strong enough to be considered a full test suite.
They serve rather as 'smoke tests', for if anything fails completely.
"""

from synapticonn import SynaptiConn
import pytest
import pathlib
import os
import pandas as pd

from tests.tutils import test_data_path, load_spiketimes


################################


@pytest.mark.parametrize("data_type", ['.mat', 'spikeinterface'])
def test_base_init(data_type):
    """ Test the SynaptiConn model object with different spike time data types. """

    # load spike times based on the data type
    spiketimes = load_spiketimes(data_type)

    # initialize the SynaptiConn model
    model = SynaptiConn(
        spike_times=spiketimes,
        bin_size_t=0.0005,
        time_unit='s',
        max_lag_t=0.20,
        srate=20_000,
        recording_length_t=1000,
    )

    # assert that the model is correctly initialized
    assert isinstance(model, SynaptiConn)
    assert model.bin_size_t == 0.0005
    assert model.time_unit == 's'
    assert model.max_lag_t == 0.20
    assert model.srate == 20_000
    assert model.recording_length_t == 1000
