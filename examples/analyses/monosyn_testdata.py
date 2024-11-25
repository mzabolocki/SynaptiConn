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

# open mat file
data = scipy.io.loadmat(data_fpath)
# get all spiketrain units and convert to milliseconds
all_units = {i: data['unit_t'][0][i].T[0] * 1000 for i in range(len(data['unit_t'][0]))}

################################################################
# Initialize the object
# ~~~~~~~~~~~~~~~~~~~~~
# Initialize the SynaptiConn object.
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
#

################################################################

snc = synapticonn.SynaptiConn(all_units,
                              method='cross-correlation',
                              bin_size_t=1,
                              time_unit='ms',
                              max_lag_t=200,
                              srate=20_000,
                              recording_length_t=1000*1000,
                              spike_id_type=int)

################################################################
#
# Now, we have initialized the SynaptiConn object. We can now proceed to the
# analysis. However, before proceeding, it is important to check the loaded
# spike times data. The spike times data should be in the correct format.
#
# to do so, SynaptiConn provides a method 'report_spike_units' to check the spike times data.
#

################################################################

spk_unit_report = snc.report_spike_units()
print(spk_unit_report)

################################################################
# Spike isolation quality metrics
# ~~~~~~~~~~~~~~~~~~~~~
#
# Before continuining, it is important to cross-check the quality of the spike sorted neurons. 
# Metrics related to the spike quality can be found below. Notably, the autocorrelograms
# for each unit should also be cross-referenced prior to continuing.
# Low contamination (or no contamination) in the refractory periods are important
# for correct assesments of spike-units and their monosynaptic connections.
#
# **NOTE** here, more simple and core metric assessments are performed.
# In the future, these will be extended. For further quality metrics,
# and explanations, please refer to the following: https://github.com/SpikeInterface/spikeinterface/blob/main/src/spikeinterface/qualitymetrics/misc_metrics.py#L1183). Further, Allen Brain have core documentation which can be found [here](https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_quality_metrics.html#ISI-violations
#

################################################################

# note :: isi min should be based on the
# miniimum possible refractory period (e.g. spikes removed would constitute this)
# isi threshold should be based on the refractory period of the neuron

params = {'isi_threshold_ms': 1.5,
          'min_isi_ms': 0,
          'presence_ratio_bin_duration_sec': 60,
          'presence_ratio_mean_fr_ratio_thresh': 0.0}

qc = snc.spike_unit_quality(**params)
qc

################################################################