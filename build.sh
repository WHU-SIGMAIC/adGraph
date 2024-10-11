#!/bin/bash

# Copyright (c) 2019, NVIDIA CORPORATION.

# nvgraph build script

# This script is used to build the component(s) in this repo from
# source, and can be called with various options to customize the
# build as needed (see the help output for details)

# Abort script on first error
set -e

NUMARGS=$#
ARGS=$*

# NOTE: ensure all dir changes are relative to the location of this
# script, and that this script resides in the repo dir!
REPODIR=$(cd $(dirname $0); pwd)

VALIDARGS="clean -v -g -n -h --help"
HELP="$0 [<target> ...] [<flag> ...]
 where <target> is:
   clean      - remove all existing build artifacts and configuration (start over)
 and <flag> is:
   -v         - verbose build mode
   -n         - no install step
   -h         - print this text

 default action (no args) is to build and install 'libnvgraph' targets
"

LIBNVGRAPH_BUILD_DIR=${REPODIR}/cpp/build
BUILD_DIRS="${LIBNVGRAPH_BUILD_DIR}"

# Set defaults for vars modified by flags to this script
VERBOSE=""
BUILD_TYPE=Release
INSTALL_TARGET=install

INSTALL_PREFIX=${DTK_ROOT}
PARALLEL_LEVEL=${PARALLEL_LEVEL:=""}
BUILD_ABI=${BUILD_ABI:=ON}

function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

if hasArg -h || hasArg --help; then
    echo "${HELP}"
    exit 0
fi

# Check for valid usage
if (( ${NUMARGS} != 0 )); then
    for a in ${ARGS}; do
  if ! (echo " ${VALIDARGS} " | grep -q " ${a} "); then
      echo "Invalid option: ${a}"
      exit 1
  fi
    done
fi

# Process flags
if hasArg -v; then
    VERBOSE=1
fi
if hasArg -g; then
    BUILD_TYPE=Debug
fi
if hasArg -n; then
    INSTALL_TARGET=""
fi

# If clean given, run it prior to any other steps
if hasArg clean; then
    # If the dirs to clean are mounted dirs in a container, the
    # contents should be removed but the mounted dirs will remain.
    # The find removes all contents but leaves the dirs, the rmdir
    # attempts to remove the dirs but can fail safely.
    for bd in ${BUILD_DIRS}; do
  if [ -d ${bd} ]; then
      find ${bd} -mindepth 1 -delete
      rmdir ${bd} || true
  fi
    done
fi

################################################################################
# Configure, build, and install libnvgraph
mkdir -p ${LIBNVGRAPH_BUILD_DIR}
cd ${LIBNVGRAPH_BUILD_DIR}
cmake -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
      -DCMAKE_CXX11_ABI= "${BUILD_ABI}" \
      -D CMAKE_HIP_COMPILER_ROCM_ROOT= "${INSTALL_PREFIX}" \
      -D CMAKE_HIP_ABI_COMPILED=Yes \
      -D CMAKE_HIP_COMPILER= "${INSTALL_PREFIX}/llvm/bin/clang-14" \
      -D CMAKE_CXX_COMPILER= "${INSTALL_PREFIX}/bin/hipcc"
      -DCMAKE_BUILD_TYPE=${BUILD_TYPE} ..
make -j${PARALLEL_LEVEL} VERBOSE=${VERBOSE} ${INSTALL_TARGET}
