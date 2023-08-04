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

Example notebooks are in ``pwg-tutorials/toast``.

.. note:: Not quite yet.  In progress and using features from PR 183.


Batch Use
-------------------

When running workflows in batch mode, we typically use a slightly different
structure from a notebook. There are helper functions in toast to parse
operator options from the commandline and / or a config file, and to
automatically create named instances of those operators for use in workflow.
There is a large-scale simulation and reduction workflow in
``sotodlib/workflows/toast_so_sim.py``. This script has nearly every possible
operator in its config options, and they can be selectively enabled or disabled
from the commandline or config file.

For processing data that already exists on disk, you can use
``sotodlib/workflows/toast_so_map.py`` as an example workflow. This loads data
in the native toast HDF5 format and / or L3 books.

.. note:: Loading data from books will be added in PR 183.


Reference
---------------------

This is an API reference for the software in ``sotodlib.toast``.


Instrument Model
^^^^^^^^^^^^^^^^^^^^^^

The TOAST instrument model consists of a ``Telescope`` instance which is
associated with a ``Site`` and a ``Focalplane``.  A particular Focalplane
instance usually contains detectors from one readout card / wafer.

.. autoclass:: sotodlib.toast.SOFocalplane
   :members:

.. note::  There are more extensive updates to the S.O. toast instrument classes coming in PR 183.


Operators
^^^^^^^^^^^^^^^^

The operators in ``sotodlib.toast.ops`` can be used in a workflow along with
the standard operators in toast.

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

.. autoclass:: sotodlib.toast.ops.Hn
   :members:
   :noindex:

.. autoclass:: sotodlib.toast.ops.MLMapmaker
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


Data I/O
^^^^^^^^^^^^^^

.. note:: Expand this during PR 183.
