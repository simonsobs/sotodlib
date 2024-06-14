.. _reference:

API Reference
========================

This package contains tools for working with hardware information (as needed
for analysis operations) as well as timestream processing, workflow management
and other operations needed for time domain data.

Hardware Properties
-----------------------

.. Note this py:module is required for autoclass to work, below.
.. py:module:: sotodlib.sim_hardware

For the purpose of this package, "hardware" refers to all properties of the telescopes, detectors, readout, etc that are needed to simulate and analyze the data.  Initially this will be a fairly basic set of information, but that will expand as the instrument characterization progresses.

Data Format
~~~~~~~~~~~~~~~~~~~

In memory, the hardware configuration is stored as a set of nested dictionaries.  This is wrapped by a simple "Hardware" class that has some methods for dumping / loading and selecting a subset of detectors.  Some parameters may eventually reference external data files in a format and location scheme that is yet to be determined.

.. autoclass:: Hardware
    :members:

To get an example hardware configuration as a starting point, you can use this function:

.. autofunction:: sotodlib.sim_hardware.sim_nominal

Simulated Detectors
~~~~~~~~~~~~~~~~~~~~~~~~

Given a Hardware object loaded from disk or created in memory (for example using :func:`sotodlib.hardware.sim_nominal`), detector properties can be simulated for an entire telescope with:

.. autofunction:: sotodlib.sim_hardware.sim_detectors_toast

OR with a more realistic method:

.. autofunction:: sotodlib.sim_hardware.sim_detectors_physical_optics

These functions update the Hardware object in place:

.. code-block:: python

    hw = sim_nominal()
    # Adds detectors to input Hardware instance.
    sim_detectors_toast(hw, "LAT")


Visualization
~~~~~~~~~~~~~~~~~~~~~~

.. py:module:: sotodlib.vis_hardware

The detectors in a given Hardware model can be plotted with this function:

.. autofunction:: plot_detectors

To plot only a subset of detectors, first apply a selection to make a reduced hardware model and pass that to the plotting function.  You can also dump out to the console a pretty formatted summary of the hardware configuration:

.. autofunction:: summary_text


Data Processing
-----------------------
These modules are used to process detector timestream data within a G3Pipeline. The base class, :class:`DataG3Module`, handles the translation between a G3TimestreamMap and any filtering/conditioning we want to do on the timestreams.

DataG3Module
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: sotodlib.core.g3_core.DataG3Module
    :members:


Filters
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sotodlib.g3_filter
   :members:

Conditioning
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sotodlib.g3_condition
   :members:
