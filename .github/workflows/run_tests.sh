#!/bin/bash

set -e

# This script runs inside our testing docker container

cp -a /home/sotodlib .

pushd sotodlib >/dev/null 2>&1

export SOTODLIB_TEST_DISABLE_PLOTS=1

python3 setup.py test

popd >/dev/null 2>&1
