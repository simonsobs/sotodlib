===========
TOAST
===========

The ``sotodlib.toast`` package contains toast operators for use in
workflows for Simons Observatory.  The first step is to make sure
you have TOAST (and sotodlib) installed.


User Installation
------------------------

If you are using a stable release, then you can install the ``toast``
package directly from either a python wheel or a conda package.  Make
sure to create a virtualenv or conda env first::

    conda install toast

OR::

    pip install toast

And follow that by installing ``so3g``::

    pip install so3g

and ``sotodlib``::

    pip install git+https://github.com/simonsobs/sotodlib.git@master

OR, if you have a local git checkout of sotodlib::

    cd sotodlib
    pip install .

.. note::  Until toast-3.0 is released, use a pre-release python wheel
    from PyPI by doing ``pip install --pre toast``.


Developer Installation
--------------------------

If you will be installing TOAST and modifying the source, then the
easiest way forward is to use a conda environment.  This is also the
path to take if you want to use the latest upstream code version.

To begin, make sure you already have a recent conda "base" environment.
If you are starting from scratch, I recommend running the "miniforge" `installer from conda-forge <https://github.com/conda-forge/miniforge>`_.
Make sure your base conda environment is activated::

    conda env list

Now decide on a name for your new development environment.  In this example
we use "simons".  Clone toast locally and checkout the toast3 branch::

    git clone https://github.com/hpc4cmb/toast.git
    cd toast
    git checkout -b toast3 -t origin/toast3

Next we will run a helper script to create (or activate) our new conda env
and install toast dependencies.  Use the name you have chosen here::

    cd toast
    ./platforms/conda_dev_setup.sh simons

This new environment should show up with::

    conda env list

Now we can install the toast package itself after activating our environment::

    cd toast
    conda activate simons
    ./platforms/conda.sh

Whenever you pull new changes to your git checkout, you can re-run the
``platforms/conda.sh`` script to install an updated version of the toast
package.  Now install ``so3g``::

    pip install so3g

and ``sotodlib``::

    pip install git+https://github.com/simonsobs/sotodlib.git@master

OR, if you have a local git checkout of sotodlib::

    cd sotodlib
    pip install .


Interactive Use
---------------------

Example notebooks will be placed in ``pwg-tutorials/toast``. You can mix and
match high-level "workflow" functions to do broad tasks (load data, simulate
observing, mapmaking, etc) and other individual operators or your own custom
code.


Batch Use
-------------------

When running workflows in batch mode, we typically use a slightly different
structure from a notebook. There is usually additional boilerplate code to
ensure that any unhandled python exceptions correctly terminate an MPI job.
There is often more extensive logging of the job configuration and also
workflow functions will get their arguments exclusively from config files and
commandline arguments.

There is a large-scale simulation and reduction workflow in
``sotodlib/toast/scripts/so_sim.py``. This script has nearly every possible
operator and focuses on a typical sequence of simulating different effects and
multiple different analyses. Different reduction paths and mapmakers can be
selectively enabled or disabled from the commandline or config files.

For processing data that already exists on disk, you can look at
``sotodlib/toast/scripts/so_map.py`` as an example workflow. This loads data in any
of the supported formats and has multiple mapmaking / reduction code paths that
can be enabled.

Of course not every script needs to include all of these simulation and
reduction techniques. You can make small, focused workflows that do only the
things you want.


Reference
---------------------

This is an API reference for the software in ``sotodlib.toast``.

Workflow Tools
^^^^^^^^^^^^^^^^^^^^^^^

For high-level operations (e.g. "load some data", "make a map", etc) it is
often helpful to leverage the workflow functions that can use a mix of config
files and user parameters to setup and run a pre-defined set of operations. The
best examples are in the source files ``sotodlib/toast/workflows/so_*.py``. The
basic pattern is to select the high level operations you want to use, call the
"setup" function for each, call the ``setup_job`` function to merge options
from config files and other sources, and then run the operations in the desired
sequence.


Here is a list of the supported high-level operations:

.. autofunction:: sotodlib.toast.workflows.setup_load_data_context
.. autofunction:: sotodlib.toast.workflows.load_data_context

.. autofunction:: sotodlib.toast.workflows.setup_load_data_hdf5
.. autofunction:: sotodlib.toast.workflows.load_data_hdf5

.. autofunction:: sotodlib.toast.workflows.setup_load_data_books
.. autofunction:: sotodlib.toast.workflows.load_data_books

.. autofunction:: sotodlib.toast.workflows.setup_save_data_hdf5
.. autofunction:: sotodlib.toast.workflows.save_data_hdf5

.. autofunction:: sotodlib.toast.workflows.setup_save_data_books
.. autofunction:: sotodlib.toast.workflows.save_data_books

.. autofunction:: sotodlib.toast.workflows.setup_pointing
.. autofunction:: sotodlib.toast.workflows.select_pointing

.. autofunction:: sotodlib.toast.workflows.setup_demodulate
.. autofunction:: sotodlib.toast.workflows.demodulate

.. autofunction:: sotodlib.toast.workflows.setup_deconvolve_detector_timeconstant
.. autofunction:: sotodlib.toast.workflows.deconvolve_detector_timeconstant

.. autofunction:: sotodlib.toast.workflows.setup_filter_hwpss
.. autofunction:: sotodlib.toast.workflows.filter_hwpss

.. autofunction:: sotodlib.toast.workflows.setup_filter_ground
.. autofunction:: sotodlib.toast.workflows.filter_ground

.. autofunction:: sotodlib.toast.workflows.setup_filter_poly1d
.. autofunction:: sotodlib.toast.workflows.filter_poly1d

.. autofunction:: sotodlib.toast.workflows.setup_filter_poly2d
.. autofunction:: sotodlib.toast.workflows.filter_poly2d

.. autofunction:: sotodlib.toast.workflows.setup_filter_common_mode
.. autofunction:: sotodlib.toast.workflows.filter_common_mode

.. autofunction:: sotodlib.toast.workflows.setup_flag_sso
.. autofunction:: sotodlib.toast.workflows.flag_sso

.. autofunction:: sotodlib.toast.workflows.setup_flag_noise_outliers
.. autofunction:: sotodlib.toast.workflows.flag_noise_outliers

.. autofunction:: sotodlib.toast.workflows.setup_mapmaker_filterbin
.. autofunction:: sotodlib.toast.workflows.mapmaker_filterbin

.. autofunction:: sotodlib.toast.workflows.setup_mapmaker_madam
.. autofunction:: sotodlib.toast.workflows.mapmaker_madam

.. autofunction:: sotodlib.toast.workflows.setup_mapmaker_ml
.. autofunction:: sotodlib.toast.workflows.mapmaker_ml

.. autofunction:: sotodlib.toast.workflows.setup_mapmaker
.. autofunction:: sotodlib.toast.workflows.mapmaker

.. autofunction:: sotodlib.toast.workflows.setup_noise_estimation
.. autofunction:: sotodlib.toast.workflows.noise_estimation

.. autofunction:: sotodlib.toast.workflows.setup_raw_statistics
.. autofunction:: sotodlib.toast.workflows.raw_statistics

.. autofunction:: sotodlib.toast.workflows.setup_filtered_statistics
.. autofunction:: sotodlib.toast.workflows.filtered_statistics

.. autofunction:: sotodlib.toast.workflows.setup_hn_map
.. autofunction:: sotodlib.toast.workflows.hn_map

.. autofunction:: sotodlib.toast.workflows.setup_cadence_map
.. autofunction:: sotodlib.toast.workflows.cadence_map

.. autofunction:: sotodlib.toast.workflows.setup_crosslinking_map
.. autofunction:: sotodlib.toast.workflows.crosslinking_map

.. autofunction:: sotodlib.toast.workflows.setup_simulate_observing
.. autofunction:: sotodlib.toast.workflows.simulate_observing

.. autofunction:: sotodlib.toast.workflows.setup_simple_noise_models
.. autofunction:: sotodlib.toast.workflows.simple_noise_models

.. autofunction:: sotodlib.toast.workflows.setup_simulate_sky_map_signal
.. autofunction:: sotodlib.toast.workflows.simulate_sky_map_signal

.. autofunction:: sotodlib.toast.workflows.setup_simulate_conviqt_signal
.. autofunction:: sotodlib.toast.workflows.simulate_conviqt_signal

.. autofunction:: sotodlib.toast.workflows.setup_simulate_source_signal
.. autofunction:: sotodlib.toast.workflows.simulate_source_signal

.. autofunction:: sotodlib.toast.workflows.setup_simulate_sso_signal
.. autofunction:: sotodlib.toast.workflows.simulate_sso_signal

.. autofunction:: sotodlib.toast.workflows.setup_simulate_catalog_signal
.. autofunction:: sotodlib.toast.workflows.simulate_catalog_signal

.. autofunction:: sotodlib.toast.workflows.setup_simulate_atmosphere_signal
.. autofunction:: sotodlib.toast.workflows.simulate_atmosphere_signal

.. autofunction:: sotodlib.toast.workflows.setup_simulate_wiregrid_signal
.. autofunction:: sotodlib.toast.workflows.simulate_wiregrid_signal

.. autofunction:: sotodlib.toast.workflows.setup_simulate_stimulator_signal
.. autofunction:: sotodlib.toast.workflows.simulate_stimulator_signal

.. autofunction:: sotodlib.toast.workflows.setup_simulate_scan_synchronous_signal
.. autofunction:: sotodlib.toast.workflows.simulate_scan_synchronous_signal

.. autofunction:: sotodlib.toast.workflows.setup_simulate_hwpss_signal
.. autofunction:: sotodlib.toast.workflows.simulate_hwpss_signal

.. autofunction:: sotodlib.toast.workflows.setup_simulate_detector_timeconstant
.. autofunction:: sotodlib.toast.workflows.simulate_detector_timeconstant

.. autofunction:: sotodlib.toast.workflows.setup_simulate_detector_noise
.. autofunction:: sotodlib.toast.workflows.simulate_detector_noise

.. autofunction:: sotodlib.toast.workflows.setup_simulate_readout_effects
.. autofunction:: sotodlib.toast.workflows.simulate_readout_effects

.. autofunction:: sotodlib.toast.workflows.setup_simulate_detector_yield
.. autofunction:: sotodlib.toast.workflows.simulate_detector_yield

.. autofunction:: sotodlib.toast.workflows.setup_simulate_calibration_error
.. autofunction:: sotodlib.toast.workflows.simulate_calibration_error

.. autofunction:: sotodlib.toast.workflows.setup_job


Simulated Instrument Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The TOAST instrument model consists of a ``Telescope`` instance which is
associated with a ``Site`` and a ``Focalplane``. These are specified for every
observation- typically a single wafer and frequency for some constant elevation
scan. When working with a "bootstrapped" instrument model that is simulated
from nominal parameters, we have use a special Focalplane class which uses the
synthetic hardware model to build the table of detector properties and
boresight offsets:

.. autoclass:: sotodlib.toast.SOFocalplane
   :members:


Instrument Model for Real Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When starting from real data, we already have most properties available
directly and associated with a readout ID. We use this to build our
``Focalplane`` detector table directly. If the detector to readout mapping is
available, the detector name is also added to this table. Similarly for the
detector focalplane offsets.


Simulation Operators
^^^^^^^^^^^^^^^^^^^^^^^

Here we document only the tools that are specific to Simons Observatory. There
are many other generic things that are supplied in toast. The operators in
``sotodlib.toast.ops`` can be used in a workflow along with the standard
operators in toast.

.. autoclass:: sotodlib.toast.ops.CoRotator
   :members:
   :noindex:

.. autoclass:: sotodlib.toast.ops.SimSSO
   :members:
   :noindex:

.. autoclass:: sotodlib.toast.ops.SimCatalog
   :members:
   :noindex:

.. autoclass:: sotodlib.toast.ops.SimHWPSS
   :members:
   :noindex:

.. autoclass:: sotodlib.toast.ops.SimSource
   :members:
   :noindex:

.. autoclass:: sotodlib.toast.ops.SimWireGrid
   :members:
   :noindex:

.. autoclass:: sotodlib.toast.ops.SimStimulator
   :members:
   :noindex:

.. autoclass:: sotodlib.toast.ops.SimReadout
   :members:
   :noindex:


Data Reduction Operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These are the data reduction operators specific to Simons Observatory. Many
other operators for different kinds of mapmaking, filtering, demodulation, etc
are available in toast.

.. autoclass:: sotodlib.toast.ops.Hn
   :members:
   :noindex:

.. autoclass:: sotodlib.toast.ops.MLMapmaker
   :members:
   :noindex:


Data Load / Save Operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The toast package has built-in operators for saving / loading data to / from a
directory of HDF5 files. There are also several operators specific to Simons
Observatory:

.. autoclass:: sotodlib.toast.ops.SaveBooks
   :members:
   :noindex:

.. autoclass:: sotodlib.toast.ops.LoadBooks
   :members:
   :noindex:

.. .. autoclass:: sotodlib.toast.ops.LoadContext
..    :members:
..    :noindex:
