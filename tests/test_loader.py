""" test_load.py

Test the spike time loader functions and checks in synapticonn.

NOTES
-----
The tests here are not strong enough to be considered a full test suite.
They serve rather as 'smoke tests', for if anything fails completely.
"""

import synapticonn
import unittest
import pathlib
import os
import pandas as pd

from tests.tutils import test_data_path, load_spiketimes


################################


class TestLoader(unittest.TestCase):
    """Test the spike time loader functions in synapticonn."""

    def setUp(self):
        """ Set up the test. """

        self.mat_spiketimes = load_spiketimes('.mat')
        self.spikeinterface_spiketimes = load_spiketimes('spikeinterface')

        # initialize the model object
        self.snc = synapticonn.SynaptiConn(self.mat_spiketimes,
                                           bin_size_t=0.0005,
                                           time_unit='s',
                                           max_lag_t=0.20,
                                           srate=20_000,
                                           recording_length_t=1000)

    def test_model_object(self):
        """ Test the model object. """

        self.assertIsInstance(self.snc, synapticonn.SynaptiConn)

