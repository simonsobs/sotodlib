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
sure to create a virtualenv or conda env first:

    conda install toast

OR

    pip install toast

And follow that by installing ``so3g`` and ``sotodlib`` as usual:

    pip install so3g
    cd sotodlib
    pip install .

.. note::  Until toast-3.0 is released, use the developer instructions


Developer Installation
--------------------------

The easiest way to install TOAST for modifying the code is to use a
conda environment.  If you do not already have a conda "base" environment,
I recommend running the "miniforge" `installer from conda-forge <https://github.com/conda-forge/miniforge>`_.
Make sure you base cond environment is activated:

    conda env list

Now decide on a name for you new development environment.  In this example
we use "simons".  Clone toast locally and checkout the toast3 branch:

    git clone https://github.com/hpc4cmb/toast.git
    cd toast
    git checkout -b toast3 -t origin/toast3

Next we will run a helper script to create (or activate) our new conda env
and install toast dependencies.

    cd toast
    ./platforms/conda_dev_setup.sh simons

Now we can install the toast package itself.  If we are installing from source
just to get a newer version of toast than is available from conda-forge or PyPI,
then we can install toast as a package into our env:

    cd toast
    conda activate simons
    ./platforms/conda.sh

Whenever you pull new changes to your git checkout, you can re-run the ``conda.sh``
script to install a new version of the toast package.  Now install ``so3g`` and ``sotodlib`` as usual:

    pip install so3g
    cd sotodlib
    pip install .


Interactive Use
---------------------

Notebooks in pwg-tutorials...


Batch Use
-------------------

Example toast_sim_so.sh workflow...


Reference
---------------------

