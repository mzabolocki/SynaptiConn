""" test_fit.py

Tests monosynaptic connection fit for the SynaptiConn model object.

NOTES
-----
The tests here are not strong enough to be considered a full test suite.
They serve rather as 'smoke tests', for if anything fails completely.
"""

import synapticonn
import pytest
import pandas as pd

from tests.tutils import load_spiketimes


###############################################


@pytest.mark.parametrize("data_type", ['.mat'])
def test_fit(data_type):
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

    # -- check the spk unit report
    spk_unit_report = model.report_spike_units()
    assert spk_unit_report['n_spikes'][0] == 3238

    # -- check the quality metrics
    params = {'isi_threshold_ms': 1.5,
              'min_isi_ms': 0,
              'presence_ratio_bin_duration_sec': 60,
              'presence_ratio_mean_fr_ratio_thresh': 0.0}

    qc = model.spike_unit_quality(**params)
    presence_ratio = round(float(qc.loc[0, 'presence_ratio']), 2)
    isi_violations_ratio = round(float(qc.loc[0, 'isi_violations_ratio']), 5)
    assert presence_ratio == 1.0
    assert isi_violations_ratio == 0.00267

    # -- model fit
    # check the number of excitatory connections detected
    # w/ set threshold and iterations

    model.set_bin_settings(bin_size_t=0.5, max_lag_t=10, time_unit='ms')
    spike_pairs = [(0, 1), (0, 2), (0, 3), (0, 5), (0, 6)]
    connection_data = model.fit(spike_pairs, synaptic_strength_threshold=5, num_iterations=1000)

    connections_df = pd.DataFrame(connection_data).T
    exc_connections = connections_df[connections_df.putative_exc_connection_type == 'excitatory monosynaptic']
    assert len(exc_connections) == 5
