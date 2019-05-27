.. _reference:

API Reference
========================

This package contains tools for working with hardware information (as needed
for analysis operations) as well as timestream processing, workflow management
and other operations needed for time domain data.

Hardware Properties
-----------------------

For the purpose of this package, "hardware" refers to all properties of the telescopes, detectors, readout, etc that are needed to simulate and analyze the data.  Initially this will be a fairly basic set of information, but that will expand as the instrument characterization progresses.

Data Format
~~~~~~~~~~~~~~~~~~~

In memory, the hardware configuration is stored as a set of nested dictionaries.  This is wrapped by a simple "Hardware" class that has some methods for dumping / loading and selecting a subset of detectors.  Some parameters may eventually reference external data files in a format and location scheme that is yet to be determined.

.. autoclass:: sotodlib.hardware.Hardware
    :members:

To get an example hardware configuration as a starting point, you can use this function:

.. autofunction:: sotodlib.hardware.get_example

Simulated Detectors
~~~~~~~~~~~~~~~~~~~~~~~~

Given a Hardware object loaded from disk or created in memory (for example using :func:`sotodlib.hardware.get_example`), detector properties can be simulated for an entire telescope with:

.. autofunction:: sotodlib.hardware.sim_telescope_detectors

The resulting detector dictionary can be used independently or can be inserted into the existing Hardware object:

.. code-block:: python

    hw = get_example()
    dets = sim_telescope_detectors(hw, "LAT")
    # Replace hardware model detectors with these:
    hw.data["detectors"] = dets
    # OR, just append these detectors:
    hw.data["detectors"].update(dets)

Visualization
~~~~~~~~~~~~~~~~~~~~~~

The detectors in a given Hardware model can be plotted with this function:

.. autofunction:: sotodlib.hardware.plot_detectors

To plot only a subset of detectors, first apply a selection to make a reduced hardware model and pass that to the plotting function.  You can also dump out to the console a pretty formatted summary of the hardware configuration:

.. autofunction:: sotodlib.hardware.summary_text


Data Processing
-----------------------
These modules are used to process detector timestream data within a G3Pipeline. The base class, :class:`DataG3Module`, handles the translation between a G3TimestreamMap and any filtering/conditioning we want to do on the timestreams.  

DataG3Module
~~~~~~~~~~~~~~~~~~~~~~   
.. autoclass:: sotodlib.data.DataG3Module
    :members:


Filters
~~~~~~~~~~~~~~~~~~~~~~   

.. automodule:: sotodlib.data.filter
   :members:
   
Conditioning
~~~~~~~~~~~~~~~~~~~~~~   

.. automodule:: sotodlib.data.condition
   :members:
