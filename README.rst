=========================
SynaptiConn
=========================

|ProjectStatus| |Version| |BuildStatus| |Coverage| |License| |PythonVersions|

.. |ProjectStatus| image:: http://www.repostatus.org/badges/latest/active.svg
   :target: https://www.repostatus.org/#active
   :alt: project status

.. |Version| image:: https://img.shields.io/pypi/v/fooof.svg
   :target: https://pypi.python.org/pypi/fooof/
   :alt: version

.. |BuildStatus| image:: https://github.com/fooof-tools/fooof/actions/workflows/build.yml/badge.svg
   :target: https://github.com/fooof-tools/fooof/actions/workflows/build.yml
   :alt: build status

.. |Coverage| image:: https://codecov.io/gh/fooof-tools/fooof/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/fooof-tools/fooof
   :alt: coverage

.. |License| image:: https://img.shields.io/pypi/l/fooof.svg
   :target: https://opensource.org/licenses/Apache-2.0
   :alt: license

.. |PythonVersions| image:: https://img.shields.io/pypi/pyversions/fooof.svg
   :target: https://pypi.python.org/pypi/fooof/
   :alt: python versions


.. image:: docs/img/synapti_conn_logo_v2.png
   :alt: SynaptiConn
   :align: center
   :width: 200px

Overview
--------
SynaptiConn is a python package for inferring monosynaptic connections from single-unit spike-train data.
The package provides a set of tools for analyzing spike-train data, including spike-train cross-correlation analysis, and for inferring monosynaptic connections using a model-based approach.
The package is designed to be user-friendly and flexible, and can be used to analyze spike-train data from a variety of experimental paradigms.

Monosynaptic connections, both excitatory and inhibitory connections, are determined with a model-based approach that fits a set of connection features to the observed spike-train cross-correlation.
Using a Bayesian model selection approach, the package can determine the most likely set of connections that underlie the observed cross-correlation.

Please Star the project to support us and Watch to always stay up-to-date!

Installation
------------

To install the stable version of SynaptiConn, you can use pip:

SynaptiConn (stable version)
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install synapticonn

The development version of SynaptiConn can be installed by cloning the repository and 
installing using pip:

Development version
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git clone https://github.com/mzabolocki/SynaptiConn
    cd synapticonn
    pip install .


Documentation
--------
The 'synapticonn' package includes a full set of code documentation.

To see the documentation for the candidate release, see
`here <https://mzabolocki.github.io/SynaptiConn/>`_.

Dependencies
------------

`synapticonn` is written in Python, and requires Python >= 3.7 to run.

It requires the following dependencies:

- `numpy <https://github.com/numpy/numpy>`_
- `scipy <https://github.com/scipy/scipy>`_ >= 0.19
- `matplotlib <https://github.com/matplotlib/matplotlib>`_ is needed to visualize data and model fits
- `pandas <https://github.com/pandas-dev/pandas>`_ is needed for exporting connection features to dataframes
- `joblib <https://github.com/joblib/joblib>`_ is needed for parallel processing
- `openpyxl <https://github.com/theorchard/openpyxl>`_ is needed for exporting connection features to excel files

We recommend using the `Anaconda <https://www.anaconda.com/distribution/>`_ distribution to manage these requirements.


.. ## References
.. 1. https://star-protocols.cell.com/protocols/3438
