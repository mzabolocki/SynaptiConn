=========================
SynaptiConn
=========================


|ProjectStatus| |Version| |BuildStatus| |Docs| |License| |PythonVersions|

.. |ProjectStatus| image:: http://www.repostatus.org/badges/latest/active.svg
   :target: https://www.repostatus.org/#active
   :alt: project status

.. |Version| image:: https://img.shields.io/pypi/v/synapticonn.svg
   :target: https://pypi.python.org/pypi/synapticonn/
   :alt: version

.. |BuildStatus| image:: https://github.com/mzabolocki/SynaptiConn/actions/workflows/build.yml/badge.svg
   :target: https://github.com/mzabolocki/SynaptiConn/actions/workflows/build.yml
   :alt: build status

.. |Docs| image:: https://github.com/mzabolocki/SynaptiConn/actions/workflows/docs.yml/badge.svg
   :target: https://github.com/mzabolocki/SynaptiConn/actions/workflows/docs.yml
   :alt: docs status

.. |License| image:: https://img.shields.io/pypi/l/synapticonn.svg
   :target: https://opensource.org/licenses/Apache-2.0
   :alt: license

.. |PythonVersions| image:: https://img.shields.io/pypi/pyversions/synapticonn.svg
   :target: https://pypi.python.org/pypi/synapticonn/
   :alt: python versions

.. image:: https://github.com/mzabolocki/SynaptiConn/blob/main/docs/img/synapti_conn_logo_v2.png
   :alt: SynaptiConn
   :align: center
   :width: 40%


Overview
--------
SynaptiConn is a python package for inferring monosynaptic connections from single-unit spike-train data.
The package provides a set of tools for analyzing spike-train data, including spike-train cross-correlation analysis, and for inferring monosynaptic connections using a model-based approach.
The package is designed to be user-friendly and flexible, and can be used to analyze spike-train data from a variety of experimental paradigms.

Monosynaptic connections, both excitatory and inhibitory connections, are determined with a model-based approach that fits a set of connection features to the observed spike-train cross-correlation.
The package can determine the most likely set of connections that underlie the observed cross-correlation. The package also provides a set of tools for visualizing the data and model fits,
and for exporting the connection features. 

In future versions, the package will include additional tools for analyzing spike-train data, and for inferring connections from other types of data, using a variety of models.

**Please Star the project to support us and Watch to always stay up-to-date!**

Installation
------------

To install the stable version of SynaptiConn, you can use pip:

SynaptiConn (stable version)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install synapticonn

The development version of SynaptiConn can be installed by cloning the repository and 
installing using pip:

Development version
~~~~~~~~~~~~~~~~~~~~~~
To get the current development version, first clone this repository:

.. code-block:: bash
    
    git clone https://github.com/mzabolocki/SynaptiConn

To install this cloned copy, move into the directory you just cloned, and run:

.. code-block:: shell

    $ pip install .

Editable Version
~~~~~~~~~~~~~~~~~~~~~~

To install an editable version, download the development version as above, and run:

.. code-block:: shell

    $ pip install -e .

Documentation
--------------
The 'synapticonn' package includes a full set of code documentation.

To see the documentation for the candidate release, see
`here <https://mzabolocki.github.io/SynaptiConn/>`_.

Dependencies
-------------

`synapticonn` is written in Python, and requires Python >= 3.7 to run.

It requires the following dependencies:

- `numpy <https://github.com/numpy/numpy>`_
- `scipy <https://github.com/scipy/scipy>`_ >= 0.19
- `matplotlib <https://github.com/matplotlib/matplotlib>`_ is needed to visualize data and model fits
- `pandas <https://github.com/pandas-dev/pandas>`_ is needed for exporting connection features to dataframes
- `joblib <https://github.com/joblib/joblib>`_ is needed for parallel processing
- `openpyxl <https://github.com/theorchard/openpyxl>`_ is needed for exporting connection features to excel files

We recommend using the `Anaconda <https://www.anaconda.com/distribution/>`_ distribution to manage these requirements.

Quick start
-----------
The module is object orientated, and the main class is `SynaptiConn`, which is used to analyze spike-train data and infer monosynaptic connections.

An example how to use the package is shown below:

.. code-block:: python
   
    # import the model object
    from synapticonn import SynaptiConn

    # initialize the model object
    snc = SynaptiConn(spike_times,
                      method="cross-correlation",
                      time_unit="ms",
                      srate=30_000,
                      recording_length_t=600*1000,
                      bin_size_t=1,
                      max_lag_t=100,)
 
    # set the spike unit ids to be used for the analysis
    spike_pairs = [(0, 6), (0, 7), (0, 8), (0, 9)]
 
    # fit the model and report the monosynaptic connection results
    snc.report(spike_pairs)

.. An example of the report output is shown below:

.. .. image:: https://github.com/mzabolocki/SynaptiConn/blob/main/docs/img/report_summary.png
..    :alt: report_summary
..    :align: center

**Define the settings**

The `SynaptiConn` object is initialized with the following settings:

- `spike_times` : dict
    A dictionary of spike times for each neuron, where the keys are the neuron IDs, and the values are arrays of spike times.
- `method` : str
      The method to use for inferring connections. Currently, only 'cross-correlation' is supported. This will be expanded in future versions.
- `time_unit` : str
      The time unit of the spike times. Currently, only 'ms' is supported. This will be expanded in future versions.
- `srate` : float
      The sampling rate of the spike times, in Hz.
- `recording_length_t` : float
      The length of the recording, in the same time unit as the spike times.
- `bin_size_t` : float
      The size of the bins to use for the cross-correlation analysis, in the same time unit as the spike times.
- `max_lag_t` : float
      The maximum lag to use for the cross-correlation analysis, in the same time unit as the spike times.

**Note that a full set of examples and tutorials are provided in the documentation.
These provide a more detailed overview of how to use the package, and how to interpret the results.**

Documentation will be maintained and updated regularly, and we welcome feedback and suggestions for improvements.

Spike-train data
-----------------
SynaptiConn is designed to work with spike-train data, which can be provided in the form of a dict of spike times for each neuron.
These are to be organised as a dictionary, where the keys are the neuron IDs, and the values are arrays of spike times.

It is recommended to use the `SpikeInterface <https://spikeinterface.readthedocs.io/en/latest/modules/sorters.html>`_ package to process, load and organize spike-train data.
All spike-units should be subject to appropriate spike-sorting procedures before being analyzed with SynaptiConn. This includes removing noise and artifacts,
and ensuring that the spike times are accurate. For further information, please see the quality control metric outline from
`Allen Brain documentation <https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_quality_metrics.html#d-prime>`_.

If unsure of the data quality, SynaptiConn has simple quality control checks built in, which can be used to filter out poor quality data.

*In future versions, we plan to include additional spike-time data types, such as NWB files, and other file formats. Further, 
we plan to include additional spike-time data loaders, to make it easier to load and organize spike-time data, along with additional quality control checks.*
