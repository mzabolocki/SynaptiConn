""" test_initialization.py

Tests for the initialization of the SynaptiConn model object.

NOTES
-----
The tests here are not strong enough to be considered a full test suite.
They serve rather as 'smoke tests', for if anything fails completely.
"""

import synapticonn
import pytest
import numpy as np

from tests.tutils import load_spiketimes


################################


@pytest.mark.parametrize("data_type", ['.mat', 'spikeinterface'])
def test_base_init(data_type):
    """ Test the SynaptiConn model object with different spike time data types. """

    # load spike times based on the data type
    spiketimes = load_spiketimes(data_type)

    # ---- test the initialization of the SynaptiConn model ----
    # a .mat file is used for the test
    # with different parameters, to cross-validate the initialization
    # of the SynaptiConn model
    if data_type == '.mat':
        model = synapticonn.SynaptiConn(
            spike_times=spiketimes,
            bin_size_t=0.0005,
            time_unit='s',
            method='cross-correlation',
            max_lag_t=0.20,
            srate=20_000,
            recording_length_t=1000,
        )

        assert isinstance(model, synapticonn.SynaptiConn)
        assert model.bin_size_t == 0.0005
        assert model.time_unit == 's'
        assert model.max_lag_t == 0.20
        assert model.srate == 20_000
        assert model.recording_length_t == 1000
        assert model.spike_id_type == int

    # a spikeinterface file is used for the test
    # with different time units and parameters to
    # check error handling
    elif data_type == 'spikeinterface':

        si_params = {'spike_times': spiketimes,
                     'bin_size_t': 1000,
                     'time_unit': 'ms',
                     'method': 'cross-correlation',
                     'max_lag_t': 0.20,
                     'srate': 30_000,
                     'recording_length_t': 600*1000,
                     'spike_id_type': np.int32}

        # bin size err
        with pytest.raises(ValueError, match="Bin size is greater than the allowed maximum"):
            model = synapticonn.SynaptiConn(**si_params)

        # time unit err
        si_params['spike_id_type'] = int
        with pytest.raises(synapticonn.utils.errors.SpikeTimesError, match="All keys in spike_times must be of type 'int'. "
                                                                           "Please verify the unit IDs or update the spike_id_type."):
            model = synapticonn.SynaptiConn(**si_params)

        # rec length err
        si_params['spike_id_type'] = np.int32
        si_params['recording_length_t'] = 600
        with pytest.raises(synapticonn.utils.errors.RecordingLengthError, match="Spike times for unit 5 exceed the recording length. Max spike time: 597457.5, "
                                                                                "Recording length: 600. Check that the recording length is correct and in ms."):
            model = synapticonn.SynaptiConn(**si_params)