.. _api_documentation:

=================
API Documentation
=================

API reference for the module.

Table of Contents
=================

.. contents::
   :local:
   :depth: 2

.. currentmodule:: synapticonn

Model Objects
-------------

Objects that manage spike-train data and fit the model to determine monosynaptic connections.

SynaptiConn Object
~~~~~~~~~~~~~~~~~~~

The SynaptiConn object is the base object for the model, and can be used to infer monosynaptic connections from individual spike-units.

.. autosummary::
   :toctree: generated/

   synapticonn.core.connections.SynaptiConn


SpikeManager Object
~~~~~~~~~~~~~~~~~~~~~~~~

The SpikeManager object is used to manage spike-train data and prepare it for analysis.

.. autosummary::
   :toctree: generated/

   synapticonn.core.connections.SpikeManager