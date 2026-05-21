#!/bin/bash

# Helper script to create a conda environment with all of the
# dependencies needed for developing SO software.  This includes
# packages needed when building some tools from source.
#
# This script is for anyone who does not want to use the "soconda"
# tools and / or does not need things like the libactpol / moby2
# packages.  This script installs as much as possible as conda
# packages, and then pip-installs the remainder.
#
# IMPORTANT NOTES:
#
#    1.  You should use a conda-forge "base" environment.  For
#        example downloaded from: https://conda-forge.org/download/
#        Make sure "conda" is in your path before running this script.
#
#    2.  If you are running on a cluster / HPC center you 
#        should set the MPICC environment variable to the 
#        name of your MPI C compiler.  For example, at NERSC
#        you would do:
#
#        MPICC=cc ./conda_dev_env.sh -e /path/to/my/env -p 3.13
#
# OTHER NOTES:
#
# If you would like to create your environments in an alternate
# directory, first load the base environment and add the
# alternate location to your conda config:
#
#    conda config --append envs_dirs path/to/envs
#
# Then run this script with the full location:
#
#    ./conda_dev_env.sh path/to/envs/my_env
#
# and then when activating the environment you can reference
# it with just "my_env".
#
# NOTE:  by default, the full path to the conda env will be
# displayed in the shell prompt.  To avoid this, edit ~/.condarc
# and add this line:
#
# env_prompt: '({name}) '
#
# Now only the basename of the environment will show up.

# Location of this script
pushd $(dirname $0) >/dev/null 2>&1
scriptdir=$(pwd)
popd >/dev/null 2>&1

show_help () {
    echo "" >&2
    echo "Usage:  $0" >&2
    echo "    [-e <environment, either name or full path>]" >&2
    echo "    [-p <python version (e.g. 3.12)>]" >&2
    echo "" >&2
    echo "    Create a conda environment for SO development." >&2
    echo "" >&2
    echo "" >&2
}

envname=""
pyversion=$(python3 --version 2>&1 | awk '{print $2}' | sed -e "s#\(.*\)\.\(.*\)\..*#\1.\2#")
extra="no"

while getopts ":e:p:x" opt; do
    case $opt in
        e)
            envname="${OPTARG}"
            ;;
        p)
            pyversion="${OPTARG}"
            ;;
        \?)
            show_help
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            show_help
            exit 1
            ;;
    esac
done

shift $((OPTIND-1))

if [ -z "${envname}" ]; then
    echo ""
    echo "You must specify the environment name or path!"
    show_help
    exit 1
fi

conda_exe=$(which conda)
if [ -z "${conda_exe}" ]; then
    echo "No conda executable found- is your base environment activated?"
    exit 1
fi

# Activate the base environment
basedir=$(dirname $(dirname "${conda_exe}"))
if [[ -e ${basedir}/etc/profile.d/conda.sh ]]; then
    source ${basedir}/etc/profile.d/conda.sh
else
    source /etc/profile.d/conda.sh
fi
conda deactivate
conda activate base

# Determine whether the environment is a name or a
# full path.
env_noslash=$(echo "${envname}" | sed -e 's/\///g')

is_path=no
if [ "${env_noslash}" != "${envname}" ]; then
    # This was a path
    is_path=yes
    env_check=""
    if [ -e "${envname}/bin/conda" ]; then
        # It already exists
        env_check="${envname}"
    fi
else
    env_check=$(conda env list | { grep "${envname} " || true; })
fi

# Determine whether we are installing MPI with conda.  This matters due
# to a bug in the ucx package (required by the MPI packages).  The ucx package
# will only install correctly if listed as an initial package during the 
# "create" command.

conda_mpi="yes"
mpi_pkgs="mpich mpi4py"
if [ -n "${MPICC}" ]; then
    conda_mpi="no"
    mpi_pkgs=""
fi

if [ -z "${mpi_pkgs}" ]; then
    initial_pkgs="python=${pyversion}"
else
    if [ "$(uname)" = "Linux" ]; then
        initial_pkgs="ucx python=${pyversion}"
    else
        initial_pkgs="python=${pyversion}"
    fi
fi

if [ -z "${env_check}" ]; then
    # Environment does not yet exist.  Create it.
    echo "Creating new environment \"${envname}\""
    if [ ${is_path} = "no" ]; then
        conda create --yes -n "${envname}"
    else
        conda create --yes -p "${envname}"
    fi
    echo "Activating environment \"${envname}\""
    conda activate "${envname}"
    echo "Setting default channel in this env to conda-forge"
    conda config --env --add channels conda-forge
    conda config --env --set channel_priority strict
else
    echo "Activating environment \"${envname}\""
    conda activate "${envname}"
fi

# Install conda packages.  First the python we are using.
echo "Installing python-${pyversion}..."
conda install --yes --update-all ${initial_pkgs}

# Next, install build tools
echo "Installing conda packages for development..."
conda install --yes --update-all \
    compilers \
    pybind11 \
    cmake \
    ninja \
    scikit-build-core \
    setuptools \
    setuptools-scm

# Reload the environment to pick up compiler environment variables
conda deactivate
conda activate "${envname}"

# The conda compiler packages make a symlink "cc", which conflicts
# with the Cray MPI compiler needed for mpi4py.  Remove this symlink.
rm -f "${CONDA_PREFIX}/bin/cc"

# Install dependencies that we have on conda-forge
echo "Installing SO packages that are on conda-forge"
# FIXME: add so3g here eventually (spt3g is already there)
conda install --yes pixell toast qpoint

installed_pkgs="$(conda list | awk '{print $1}')"

# Some of our dependencies are only available from pypi.  Install
# pipgrip so that we can try to install as many dependencies as
# possible from conda-forge
python -m pip install pipgrip

pushd "${scriptdir}" 2>&1 >/dev/null

# FIXME: remove so3g (and the loop) once that package is on conda-forge
for pkg in "so3g" "."; do
    deps=$(pipgrip --threads 4 --pipe ${pkg} | sed -e 's/\(==[^[:space:]]\+\)//g')
    for dep in ${deps}; do
        if [ ${dep} != ${pkg} ]; then
            # Special handling of ruamel.yaml / ruamel-yaml.  This is a longstanding
            # mess across PyPI and conda ecosystems...
            if [ "${dep}" = "ruamel-yaml" ] || [ "${dep}" = "ruamel-yaml-clib" ]; then
                dep="ruamel.yaml"
            fi
            depcheck=$(echo "$installed_pkgs" | grep -E "^${dep}\$")
            if [ -z "${depcheck}" ]; then
                # It is not already installed, try to install it with conda
                echo "Attempt to install conda package for dependency \"${dep}\"..."
                conda install --yes --quiet ${dep}
                # A failure of the above command is NOT AN ERROR.  If the
                # conda package does not exist, then the dependency will be
                # installed by pip below.
                if [[ $? = 0 ]]; then
                    installed_pkgs="${installed_pkgs}"$'\n'"${dep}"
                fi
            else
                echo "  Package for dependency \"${dep}\" already installed"
            fi
        fi
    done
    echo "Installing package ${pkg}"
    python3 -m pip install ${pkg}
    [[ $? != 0 ]] && exit 1
    installed_pkgs="${installed_pkgs}"$'\n'"${pkg}"
done

popd 2>&1 >/dev/null

# Install mpi4py from source if using an external MPI
if [ "${conda_mpi}" = "no" ]; then
    echo "Building mpi4py with MPICC=\"${MPICC}\""
    pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
    [[ $? != 0 ]] && exit 1
else
    conda install --yes ${mpi_pkgs}
fi

