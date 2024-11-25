"""
Monosynaptic pair analysis demo
===============================

An example analysis applied to single-unit spiketrain data, demonstrating
best practices for monosynaptic pair analysis with the 'synapticonn' package.
"""

#############################################################
# Monosynaptic connection analysis demo
# -------------------------------
#
# This example demonstrates how to perform a monosynaptic connection analysis.
#
# Here, we use an existing published dataset of single-unit spiketrains
# recorded from the primary visual cortex of an anesthetized mouse.
# The dataset is available via the link: 10.1016/j.celrep.2023.113475
#

#############################################################

# Import necessary modules
import os
import pathlib
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.chdir('../..')
import synapticonn

#############################################################
# Load spike times data
# ~~~~~~~~~~~~~~~~~~~~~
#
# First, load your spike times data. Here, we load the spike times data from a
# .mat file. The spike times data should be a list of spike times for each
# neuron in the dataset. The spike times should be in seconds or milliseconds.
#

#############################################################

data_fpath = pathlib.Path('examples', 'analyses', 'data', 'all_unit.mat')

# check if file exists
if not data_fpath.exists():
    raise FileNotFoundError(f"File not found: {data_fpath}")

# open mat file
data = scipy.io.loadmat(data_fpath)

# re-organize data
num_units = len(data['unit_t'][0])
all_units = {}
for i in range(num_units):
    all_units[i] = data['unit_t'][0][i].T[0] * 1000

################################################################
# Initialize the object
# ~~~~~~~~~~~~~~~~~~~~~
# Initialize the MonosynapticPairAnalysis object with the spike times data.
#
# Before doing so, it is important to ensure your spike times data is in the
# correct format. The spike times data should be a dictionary where the keys
# are the unit IDs and the values are the spike times for each unit.
#
# The spike times should be in seconds or milliseconds. Please change the time_unit
# accordingly to relfect this.The sampling rate should be specified in Hz, and is
# essential for the analysis. The recording length should also be specified in seconds or milliseconds.
#
# Currently, the method 'cross-correlation' is supported for monosynaptic pair analysis. In this method,
# the cross-correlation between the spike trains is computed. The bin size for the cross-correlation
# should be specified in the time unit used for the spike times data. In future versions of the package,
# more methods will be supported.


snc = synapticonn.SynaptiConn(all_units,
                              method='cross-correlation',
                              bin_size_t=1,
                              time_unit='ms',
                              max_lag_t=200,
                              srate=20_000,
                              recording_length_t=1000*1000,
                              spike_id_type=int)

################################################################