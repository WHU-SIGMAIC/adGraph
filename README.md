# adGraph - Graph library on AMD-like GPUs, ported from nvGRAPH on AMD-like GPUs 

Data analytics is a growing application of high-performance computing. Many advanced data analytics problems can be couched as graph problems. In turn, many of the common graph problems today can be couched as sparse linear algebra. This is the motivation for adGRAPH, which harnesses the power of GPUs for linear algebra to handle large graph analytics.

This repository contains the legacy version of adGRAPH. The aim is to provide a way for adGRAPH, ported version of nvGRAPH. 

## Get adGrpah
#### Prerequisites

Compiler requirement:

* `gcc`     version 5.4+
* `hipcc`    version 14
* `cmake`   version 3.12



DTK requirement:

* DTK 22.04.2+


### Using the script

It is easy to install adGRAPH from source. As a convenience, a `build.sh` script is provided. Run the script as shown below to download the source code, build and install the library.  Note that the library will be installed to the location set in `$DTK_ROOT` (eg. `export DTK_ROOT=/usr/local/dtk`). These instructions were tested on Ubuntu 18.04.

  ```bash
  git clone https://github.com/WHU-SIGMAIC/adGraph
  cd adGRAPH
  export DTK_ROOT=/usr/local/dtk
  ./build.sh  # build the adGRAPH library and install it to $DTK_ROOT (you may need to add the sudo prefix)
  ```


### Manually build from Source 

The following instructions are for developers and contributors to adGRAPH development. These instructions were tested on Linux Ubuntu 18.04. Use these instructions to build adGRAPH from source and contribute to its development.  Other operating systems may be compatible, but are not currently tested.

The adGRAPH package is a C/C++ HIP library. It needs to be installed in order for adGRAPH to operate correctly.  

The following instructions are tested on Linux systems.

#### Build and Install the C/C++ HIP components

To install adGRAPH from source, ensure the dependencies are met and follow the steps below:

1) Clone the repository and submodules

  ```bash
  # Set the localtion to adGRAPH in an environment variable adGRAPH_HOME 
  export adGRAPH_HOME=$(pwd)/adGRAPH

  # Download the adGRAPH repo
  git clone https://anonymous.4open.science/r/adGRAPH-E319.git $adGRAPH_HOME
  ```

2) Build and install `libadGRAPH_rapids.so`. CMake depends on the `hipcc` executable being on your path or defined in `$HIPCXX`.

  This project uses cmake for building the C/C++ library. To configure cmake, run:

  ```bash
  cd $adGRAPH_HOME
  cd cpp	# enter adGRAPH's cpp directory
  mkdir build   		# create build directory 
  cd build     		# enter the build directory
  cmake .. -DCMAKE_INSTALL_PREFIX="${DTK_ROOT}" \
      -DCMAKE_CXX11_ABI= "${BUILD_ABI}" \
      -D CMAKE_HIP_COMPILER_ROCM_ROOT= "${DTK_ROOT}" \
      -D CMAKE_HIP_ABI_COMPILED=Yes \
      -D CMAKE_HIP_COMPILER= "${DTK_ROOT}/llvm/bin/clang-14" \
      -D CMAKE_CXX_COMPILER= "${DTK_ROOT}/bin/hipcc"
      -DCMAKE_BUILD_TYPE=${BUILD_TYPE}

  # now build the code
  make -j				# "-j" starts multiple threads
  make install		# install the libraries 
  ```

The default installation  locations are `$CMAKE_INSTALL_PREFIX/lib` and `$CMAKE_INSTALL_PREFIX/include/adGRAPH` respectively.

