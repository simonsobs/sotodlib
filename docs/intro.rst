.. _intro:

Introduction
==============================

This package contains software for time-domain data processing.


Installation
===============

This package is currently only python with no compiled code.  It can be
installed / used with typical setuptools commands.

External Dependencies
------------------------

This code depends on a standard Python3 software stack with the usual packages
(numpy, scipy, matplotlib, etc).  There are also several additional python
package dependencies (toml, quaternionarray) that can be pip-installed.  There
are multiple ways of installing a working python3 stack on both Linux and OS X.
The solution you choose likely depends on what other things you are using
Python for- not just Simons Observatory work.  In these examples, we'll be
creating a python stack in ${HOME}/software/so, however if you already have a
python stack for use with S.O. tools, just skip this section.

Use Anaconda...
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After installing
the main Anaconda distribution, the "conda" command should be available.
Create a new conda environment::

  %> conda create --copy -m -p ${HOME}/software/so
  %> conda activate ~/software/so
  %> conda install pip numpy scipy matplotlib
  %> pip install quaternionarray

... Or Use Virtualenv and Pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install Python3 and the virtualenv package with your OS package manager (Linux)
or using macports / homebrew (OS X).  Next create a virtualenv and activate
it::

  %> virtualenv -p python3 ${HOME}/software/so
  %> source ${HOME}/software/so/bin/activate

Now use pip to install the dependencies we need::

    %> pip install numpy scipy matplotlib
    %> pip install quaternionarray


S.O. Affiliated Dependencies
---------------------------------

Activate / load your python stack from the previous section.  Since you created
a conda environment or virtualenv directory specifically for S.O. tools, you
can always delete that directory and make a new one as needed.

Currently the sotodlib package does not require any other S.O. packages.  In
the future, it will require sotoddb and so3g as dependencies.


Installing sotodlib
-----------------------------

You can either install directly to your conda environment / virtualenv::

    %> cd sotodlib
    %> python setup.py clean
    %> python setup.py install

Or (if you are frequently hacking on this code) you can install the package in
"develop" mode, which installs symlinks from your conda environment /
virtualenv that point back to your source checkout::

    %> cd sotodlib
    %> python setup.py develop


Running Tests
------------------

After installing, the unit tests can be run with::

    %> python setup.py test

Beware that these will take several minutes.


Something Went Wrong!
---------------------------

If something gets messed up, it's ok.  You are using a separate conda / virtualenv environment so you can just do::

    %> rm -rf ${HOME}/software/so

and start over from the beginning.  If you encounter an installation problem, please open a github ticket and provide the following details:

- OS and version.

- Where you got your python (Anaconda, OS packages, homebrew, etc).

- Python version (:code:`python --version`).
