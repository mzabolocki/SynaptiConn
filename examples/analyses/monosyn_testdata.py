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
#
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
# Spike-unit quality metrics
# ~~~~~~~~~~~~~~~~~~~~~
#
# Before continuining, it is important to cross-check the quality of the spike sorted neurons. 
# Metrics related to the spike quality can be found below. Notably, the autocorrelograms
# for each unit should also be cross-referenced prior to continuing.
# Low contamination (or no contamination) in the refractory periods are important
# for correct assesments of spike-units and their monosynaptic connections.
#
# **NOTE** here, more simple and core metric assessments are performed.
# In the future, these will be extended.
#
# For further quality metrics, and explanations, please refer to the following: https://github.com/SpikeInterface/spikeinterface/blob/main/src/spikeinterface/qualitymetrics/misc_metrics.py#L1183.
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
#
# Filter the spike times for 'good units' based on the quality control metrics. These will then be used for all further processing. The spike times will be updated accordingly. A log of the excluded units can be found and kept.
#
################################################################

query = 'firing_rate_hz > 0.5'
good_units = snc.filter_spike_units(qc, query, log=True)
good_units

################################################################
# Auto-correlograms
# ~~~~~~~~~~~~~~~~~~~~~
#
# Before proceeding with the monosynaptic pair analysis, visualize
# the auto-correlograms for each unit. This is important to check for
# contamination in the refractory period.
#

################################################################

snc.plot_autocorrelogram(spike_units=[1, 8], color='blue')

################################################################
#
# Set the bin parameters after initialization, and re-plot.
#
# This can be used to change the binning on the plots, and max time lags.
#

################################################################

snc.set_bin_settings(bin_size_t=0.5, max_lag_t=10, time_unit='ms')
snc.plot_autocorrelogram(spike_units=[0, 1, 8, 10], color='blue', figsize=(20, 5))

################################################################
# Cross-correlograms
# ~~~~~~~~~~~~~~~~~~~~~
#
# Visualize cross-correlograms between pairs.
#
# Bin size and time lag can be changed by re-setting the bins.
# However, for improved visualizations and reporting a smaller bin size and time lag is recommended.
#

################################################################

spk_units = snc.spike_times.keys()

# make a list of all possible pairs
pairs = []
for i, unit1 in enumerate(spk_units):
    for j, unit2 in enumerate(spk_units):
        if i < j:
            pairs.append((unit1, unit2))

# subselect for select pairs
spike_pairs = pairs[4:8]

# plot cross-correlograms
snc.plot_crosscorrelogram(spike_pairs=spike_pairs, figsize=(20, 4))

################################################################
# Cross-correlogram data
# ~~~~~~~~~~~~~~~~~~~~~
#
# Next, try returning the correlogram data.
#
# Each key in 'cross_correlations' is indexed by the unit pairs.
# The corresponding numbers refer to the spike counts, per bin.
#
# Each key in 'bins' is also indexed by the unit pairs.
# The corresponding numbers refer to the bins edges.
#

################################################################

correlogram_data = snc.return_crosscorrelogram_data(spike_pairs=spike_pairs)
correlogram_data

################################################################
#
# Check the bin settings using for correlogram generations.
#

################################################################

snc.report_correlogram_settings()

################################################################
# Compute monosynaptic connections
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Compute excitatory and inhibitory monosynaptic connections between spike trains. 
#
# This analysis was based on the following reference by Najafi.
# A link to the paper can be found here: https://doi.org/10.1016/j.neuron.2019.09.045.
#
# This protocol was based on data and experimental analyses provided in the paper, found here: 10.1016/j.celrep.2023.113475.
#
# **Computational strength calculations notes**
#
# - `First, compute synaptic strength for a set of neuron IDs. If a given unit consistently fires after a second unit, indicated by a peak in the CCG, there is high chance that these cells are functionally linked either directly through an excitatory synaptic connection or indirectly through a third neuron providing a common input.`
# - `To compute synaptic strength, the firing of a single unit in a pair was jittered across a number of iterations (num_iterations) within a time range (jitter_range_ms).`
# - `These were used to calculate a confidence interval (CI) between 1% and 99%. If the real CCG peak passed the 99% CI, the corresponding functional connection would be considered significant and not random.`
# - `A z-score was then performed using the following equation: `Z = x_real - mean_jitter / std_jitter``
#
#
# Note that the output contains the following keys:
# - 1. ccg bins
# - 2. ccg counts (from original spike trains)
# - 3. ccg counts (post jitter)
# - 4. synaptic strength
# - 5. high confidence interval (99%), calculated on jittered ccg
# - 6. low confidence interval (1%), calculation on jittered ccg
# - 7. ccg counts (within jitter range window)
# - 8. low confidence interal (1%), within jitter range window
# - 9. high confidence interal (99%), within jitter range window
#

################################################################

synaptic_strength_data = snc.synaptic_strength(spike_pairs=spike_pairs,
                                               num_iterations=1000,
                                               jitter_range_t=10)

# isolate single neuron pair
pair = (0, 6)
synaptic_strength_data[pair]

################################################################
#
# Check the synaptic strength data for a select pair.
#
# This can be done automatically by plotting the original ccg, and the z-scored value.
#

################################################################

snc.plot_synaptic_strength(spike_pair=(0,6))

################################################################
#
# **Next, check the connection type.** 
#
# Here, we can perform a putative detection using the z-score (synaptic strength) output.
#
# Thresholds should be set as > 5 for excitatory-connections, or inhibitory connections
# as < -5 based on the reference protocol.
#

################################################################

exc_connection_types = snc.monosynaptic_connection_types(synaptic_strength_threshold=5)
exc_df = pd.DataFrame(exc_connection_types).T
exc_df

################################################################
#
# Output a features dataframe containing selected spike pair connections and associated ccg features.
#
# These can be used to provide simple information on the quality of the CCG, and associated connection types.
#

################################################################

synaptic_features = snc.monosynaptic_connection_features()
synaptic_features_df = pd.DataFrame(synaptic_features).T
synaptic_features_df

################################################################
#
# The output dataframes can be merged for simplicity and further analyses.
#

################################################################

merged_df = exc_df.join(synaptic_features_df)
merged_df

################################################################
# Fit & report
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Alternatively, there is a simpler method to compute monosynaptic connections. Simply, monosynaptic connections can be inferred in one line. This combines the period methods.
#
# For feature extractions, this can be performed separately and combined.
#

################################################################

connections = snc.fit(spike_pairs, synaptic_strength_threshold=5, num_iterations=1000)
connections = pd.DataFrame(connections).T
connections

################################################################
#
# Summarize the data outputs using report. This is a convience method that calls a series of methods:
# - `fit()`
# - `print_results()`
#
# Each of these methods can be used individually.
#

################################################################

snc.report(spike_pairs, synaptic_strength_threshold=5, num_iterations=1000)

################################################################
# Monosnyaptic connection quality control
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Next, check the results. It is important to ensure the connections are truely monosynaptic, and not polysynaptic or simply a byproduct of poorly isolated spike-units.
#
# To do so, there are several in-built modules to check the quality of the outputs. These are based on the type of connections and time-lag threshold estimates.
#
# **Acceptance criteria**
# - `1. Excitatory connections between cells located near each other should occur within 2 ms.`
# - `2. The monosynaptic peak should also be shaped as a double expontential function with a fast rise and a slower decay. Inhibitory connections have a slower decay, and should be factored into your QC metrics.`
#
# **Rejection criteria**
# - `1. If the CCG shows a maintained refractory period, it suggests that the spikes should have been merged in the spike-sorting process. Hence, if a monosynaptic peak is seen, then it is likely because it is the same neuron which has not been correctly merged.`
# - `2. If the CCG peak coincides with the ACG peak (usually slower than 2 ms), the unit likely should have been merged in the spike sorting process, of the cell-pair is probably contaminated with a 3rd pair.`
# - `3. A broad centrally aligned CCG peak indicates common drive, and therefore should be rejected. This is often seen when comparing two cells located at different shanks (> hundreds of um apart). This can be difficult to differentiate.`
#

################################################################

# simply, a way to filter the connections based on the peak time of the ccg
# is to convert the connection data to a dataframe and then query it using pandas

query = 'ccg_peak_time_ms > 1 & ccg_peak_time_ms < 4'
connection_df_filtered = connections.query(query)

print(f'Number of exc connections: {len(connections)}')
print(f'Number of exc connections after filtering: {len(connection_df_filtered)}')

################################################################
#
# Beyond a simply df query, the main object can be filtered for units to be rejected. Here, a log can be 
# provided.
#

################################################################

snc.filter_connections(connections, query, log=True, overwrite=True)

################################################################
# Conclusion
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# For more information on the methods and classes used in this example, please refer to the documentation.