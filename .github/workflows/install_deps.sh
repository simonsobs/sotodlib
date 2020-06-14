#!/bin/bash

set -e

# Upgrade pip, so that we can install newer wheels

python3 -m pip install --upgrade pip

# Install external dependencies
# We upgrade scipy to fix an incompatibility with numpy

python3 -m pip install --upgrade scipy
python3 -m pip install nose toml mpi4py toast-cmb quaternionarray

# Install S.O. dependencies

python3 -m pip install https://github.com/simonsobs/sotoddb/archive/master.zip
