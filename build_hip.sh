#!/bin/bash

# Copyright (c) 2019, NVIDIA CORPORATION.
# Modified for HIP/ROCm support

# adGraph HIP build script

# This script is used to build the component(s) in this repo from
# source using HIP/ROCm, and can be called with various options to customize the
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

 default action (no args) is to build and install 'libnvgraph' targets with HIP/ROCm
"

LIBNVGRAPH_BUILD_DIR=${REPODIR}/cpp/build
BUILD_DIRS="${LIBNVGRAPH_BUILD_DIR}"

# Set defaults for vars modified by flags to this script
VERBOSE=""
BUILD_TYPE=Release
INSTALL_TARGET=install

INSTALL_PREFIX=${REPODIR}/dist
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

# Check for required environment variables
if [ -z "$DTK_ROOT" ]; then
    echo "Error: DTK_ROOT environment variable is not set"
    echo "Please set DTK_ROOT to your ROCm installation path"
    echo "Example: export DTK_ROOT=/opt/rocm"
    exit 1
fi

if [ ! -f "$DTK_ROOT/bin/hipcc" ]; then
    echo "Error: hipcc not found at $DTK_ROOT/bin/hipcc"
    echo "Please ensure ROCm is properly installed"
    exit 1
fi

echo "Building with HIP/ROCm"
echo "DTK_ROOT: $DTK_ROOT"
echo "Build type: $BUILD_TYPE"

################################################################################
# Configure, build, and install libnvgraph
mkdir -p ${LIBNVGRAPH_BUILD_DIR}
cd ${LIBNVGRAPH_BUILD_DIR}
# cmake -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
#       -DCMAKE_CXX11_ABI="${BUILD_ABI}" \
#       -DCMAKE_HIP_COMPILER_ROCM_ROOT="${DTK_ROOT}" \
#       -DCMAKE_HIP_ABI_COMPILED=Yes \
#       -DCMAKE_HIP_COMPILER="${DTK_ROOT}/llvm/bin/clang-14" \
#       -DCMAKE_CXX_COMPILER="${DTK_ROOT}/bin/hipcc" \
#       -DTHRUST_IGNORE_CUB_VERSION_CHECK=ON \
#       -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
#       ..
cmake -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
      -DCMAKE_CXX11_ABI="${BUILD_ABI}" \
      -DCMAKE_CXX_COMPILER="${DTK_ROOT}/bin/hipcc" \
      -DTHRUST_IGNORE_CUB_VERSION_CHECK=ON \
      -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
      -DBUILD_GMOCK=ON \
      ..

echo -e "\n===== =====\nBuilding libnvgraph\n===== =====\n"
echo "Install target: ${INSTALL_TARGET}"

make -j${PARALLEL_LEVEL} VERBOSE=${VERBOSE} ${INSTALL_TARGET}

echo "Build completed successfully!"
echo "Library installed to: ${INSTALL_PREFIX}"
