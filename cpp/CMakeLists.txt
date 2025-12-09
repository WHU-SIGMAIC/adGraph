#=============================================================================
# Copyright (c) 2019, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(NV_GRAPH VERSION 0.4.0 LANGUAGES C CXX HIP)

set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/arnoldi.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/kmeans.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/nvgraph.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/bfs.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/graph_contraction/contraction_mv_double_mul.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/graph_contraction/contraction_mv_double_min.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/graph_contraction/contraction_csr_max.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/graph_contraction/contraction_csr_sum.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/graph_contraction/contraction_mv_double_sum.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/graph_contraction/contraction_mv_float_min.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/graph_contraction/contraction_csr_min.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/graph_contraction/contraction_mv_double_max.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/graph_contraction/contraction_mv_float_max.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/graph_contraction/contraction_mv_float_sum.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/graph_contraction/contraction_csr_mul.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/graph_contraction/contraction_mv_float_mul.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/bfs_kernels.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/jaccard_gpu.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/pagerank.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/triangles_counting_kernels.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/kmeans.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/widest_path.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/bfs2d.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/convert.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/sssp.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/pagerank_kernels.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/nvgraph_vector_kernels.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/nvgraph_lapack.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/lobpcg.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/csrmv_cub.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/graph_extractor.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/csrmv.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/modularity_maximization.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/arnoldi.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/nvgraph_error.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/nvgraph.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/partition.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/matrix.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/size2_selector.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/lanczos.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cnmem/tests/cnmem_kernel_test.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/test_device_segmented_sort.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/test_iterator_deprecated.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/test_device_batch_memcpy.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/test_device_adjacent_difference.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/catch2_test_block_adjacent_difference.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/test_device_reduce.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/catch2_test_warp_scan.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/test_block_radix_rank.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/test_namespace_wrapped.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/test_device_histogram.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/test_device_batch_copy.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/test_iterator.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/catch2_runner.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/catch2_test_block_load.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/catch2_test_util_type.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/test_grid_barrier.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/catch2_test_cdp_wrapper.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/test_thread_sort.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/test_thread_operators.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/catch2_test_block_run_length_decode.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/catch2_test_warp_mask.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/catch2_test_printing.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/test_temporary_storage_layout.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/catch2_test_block_reduce.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/catch2_test_block_merge_sort.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/test_device_select_unique.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/test_cdp_variant_state.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/catch2_test_warp_load.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/test_device_radix_sort.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/test_device_spmv.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/link_b.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/catch2_test_warp_merge_sort.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/catch2_test_warp_reduce.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/catch2_test_warp_exchange.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/catch2_test_warp_store.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/catch2_test_block_histogram.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/catch2_test_block_store.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/test_allocator.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/test_device_select_unique_by_key.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/test_device_select_if.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/test_device_three_way_partition.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/catch2_test_block_radix_sort.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/link_a.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/test_device_scan_by_key.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/test_device_reduce_by_key.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/test_device_run_length_encode.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/test_device_scan.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/c2h/generators.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/catch2_test_block_scan.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/catch2_test_block_shuffle.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/test/test_device_merge_sort.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/examples/block/example_block_radix_sort.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/examples/block/example_block_reduce_dyn_smem.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/examples/block/example_block_reduce.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/examples/block/example_block_scan.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/examples/device/example_device_scan.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/examples/device/example_device_reduce.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/examples/device/example_device_select_flagged.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/examples/device/example_device_partition_if.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/examples/device/example_device_sort_find_non_trivial_runs.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/examples/device/example_device_radix_sort.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/examples/device/example_device_select_if.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/examples/device/example_device_partition_flagged.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/examples/device/example_device_select_unique.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub/examples/cmake/add_subdir/dummy.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/include/app/nvlouvain_app.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/include/app/nvlouvain_sample.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/include/app/nvlouvain_app_hierarchy.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/include/app/nvlouvain_sample_hierarchy.cu
           PROPERTY LANGUAGE HIP)

# Enable HIP language support
# enable_language(CUDA)
enable_language(HIP)

find_package(HIP REQUIRED)

###################################################################################################
# - compiler options ------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_HIP_STANDARD 14)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")  # -Werror")
endif(CMAKE_COMPILER_IS_GNUCXX)

set(GPU_ARCHS "" CACHE STRING
  "List of GPU architectures (semicolon-separated) to be compiled for.
   Default is to compile for a pre-determined st of gpu-architectures based
   on CTK version.")

if("${GPU_ARCHS}" STREQUAL "")
  # Default ROCm GPU architectures
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --offload-arch=gfx906 --offload-arch=gfx908 --offload-arch=gfx90a")
else()
  foreach(arch ${GPU_ARCHS})
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --offload-arch=${arch}")
  endforeach()
endif()

# set warnings as errors
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Wall")  # -Werror")

# set default build type
set(CMAKE_BUILD_TYPE "Release")

option(BUILD_TESTS "Configure CMake to build tests"
       ON)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")  # -Werror")

    option(CMAKE_CXX11_ABI "Enable the GLIBCXX11 ABI" ON)
    if(CMAKE_CXX11_ABI)
        message(STATUS "nvGraph: Enabling the GLIBCXX11 ABI")
    else()
        message(STATUS "nvGraph: Disabling the GLIBCXX11 ABI")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    endif(CMAKE_CXX11_ABI)
endif(CMAKE_COMPILER_IS_GNUCXX)

if(NOT DEFINED NVGRAPH_LIGHT)
    set(NVGRAPH_LIGHT True)
endif(NOT DEFINED NVGRAPH_LIGHT)

###################################################################################################
# - lapack modules ---------------------------------------------------------------------------------

set(LAPACK_INSTALL_ROOT "$ENV{HOME}/projects/lapack-3.12.0/install")

###################################################################################################
# - cmake modules ---------------------------------------------------------------------------------

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})

include(FeatureSummary)
include(CheckIncludeFiles)
include(CheckLibraryExists)

###################################################################################################
# - add gtest -------------------------------------------------------------------------------------

if(BUILD_TESTS)
    include(CTest)
    include(ConfigureGoogleTest)

    if(GTEST_FOUND)
        message(STATUS "Google C++ Testing Framework (Google Test) found in ${GTEST_ROOT}")
        include_directories(${GTEST_INCLUDE_DIR})
        add_subdirectory(${CMAKE_SOURCE_DIR}/tests)
    else()
        message(AUTHOR_WARNING "Google C++ Testing Framework (Google Test) not found: automated tests are disabled.")
    endif(GTEST_FOUND)
endif(BUILD_TESTS)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories(
                    "${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/thirdparty/cnmem/include"
                    "${CMAKE_SOURCE_DIR}/../external"
                    "${CMAKE_SOURCE_DIR}/../external/cusp"
                    "$ENV{DTK_ROOT}/include"
                    "$ENV{DTK_ROOT}/include/hip"
                    "${LAPACK_INSTALL_ROOT}/include"
                   )

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("$ENV{DTK_ROOT}/lib"
                 "$ENV{DTK_ROOT}/lib64"
                 "${CMAKE_BINARY_DIR}/lib"
                 "${GTEST_LIBRARY_DIR}"
                 "${LAPACK_INSTALL_ROOT}/lib64"
                 )

###################################################################################################
# - library targets -------------------------------------------------------------------------------
# if(NVGRAPH_LIGHT MATCHES True)
#     add_library(nvgraph SHARED
#                 thirdparty/cnmem/src/cnmem.cpp
#                 src/arnoldi.cu
#                 src/bfs.cu
#                 src/bfs2d.cu
#                 src/bfs_kernels.cu
#                 src/convert.cu
#                 src/csrmv.cu
#                 src/csrmv_cub.cu
#                 src/csr_graph.cpp
#                 src/graph_extractor.cu
#                 src/jaccard_gpu.cu
#                 src/kmeans.cu
#                 src/lanczos.cu
#                 src/lobpcg.cu
#                 src/matrix.cu
#                 src/modularity_maximization.cu
#                 src/nvgraph.cu
#                 src/nvgraph_cusparse.cpp
#                 src/nvgraph_cublas.cpp
#                 src/nvgraph_error.cu
#                 src/nvgraph_lapack.cu
#                 src/nvgraph_vector_kernels.cu
#                 src/pagerank.cu
#                 src/pagerank_kernels.cu
#                 src/partition.cu
#                 src/size2_selector.cu
#                 src/sssp.cu
#                 src/triangles_counting.cpp
#                 src/triangles_counting_kernels.cu
#                 src/valued_csr_graph.cpp
#                 src/widest_path.cu
#                )
# else(NVGRAPH_LIGHT MATCHES True)
#     add_library(nvgraph SHARED
#             thirdparty/cnmem/src/cnmem.cpp
#             src/arnoldi.cu
#             src/bfs.cu
#             src/bfs2d.cu
#             src/bfs_kernels.cu
#             src/convert.cu
#             src/csrmv.cu
#             src/csrmv_cub.cu
#             src/csr_graph.cpp
#             src/graph_extractor.cu
#             src/jaccard_gpu.cu
#             src/kmeans.cu
#             src/lanczos.cu
#             src/lobpcg.cu
#             src/matrix.cu
#             src/modularity_maximization.cu
#             src/nvgraph.cu
#             src/nvgraph_cusparse.cpp
#             src/nvgraph_cublas.cpp
#             src/nvgraph_error.cu
#             src/nvgraph_lapack.cu
#             src/nvgraph_vector_kernels.cu
#             src/pagerank.cu
#             src/pagerank_kernels.cu
#             src/partition.cu
#             src/size2_selector.cu
#             src/sssp.cu
#             src/triangles_counting.cpp
#             src/triangles_counting_kernels.cu
#             src/valued_csr_graph.cpp
#             src/widest_path.cu
#             src/graph_contraction/contraction_csr_max.cu
#             src/graph_contraction/contraction_csr_sum.cu
#             src/graph_contraction/contraction_mv_double_mul.cu
#             src/graph_contraction/contraction_mv_float_min.cu
#             src/graph_contraction/contraction_csr_min.cu
#             src/graph_contraction/contraction_mv_double_max.cu
#             src/graph_contraction/contraction_mv_double_sum.cu
#             src/graph_contraction/contraction_mv_float_mul.cu
#             src/graph_contraction/contraction_csr_mul.cu
#             src/graph_contraction/contraction_mv_double_min.cu
#             src/graph_contraction/contraction_mv_float_max.cu
#             src/graph_contraction/contraction_mv_float_sum.cu
#             )
# endif(NVGRAPH_LIGHT MATCHES True)

add_library(nvgraph SHARED
            thirdparty/cnmem/src/cnmem.cpp
            src/arnoldi.cu
            src/bfs.cu
            src/bfs2d.cu
            src/bfs_kernels.cu
            src/convert.cu
            src/csrmv.cu
            src/csr_graph.cpp
            src/csrmv_cub.cu
            src/graph_extractor.cu
            src/jaccard_gpu.cu
            src/kmeans.cu
            src/lanczos.cu
            src/lobpcg.cu
            src/matrix.cu
            src/modularity_maximization.cu
            src/nvgraph.cu
            src/nvgraph_cublas.cpp
            src/nvgraph_cusparse.cpp
            src/nvgraph_error.cu
            src/nvgraph_lapack.cu
            src/nvgraph_vector_kernels.cu
            src/size2_selector.cu
            src/sssp.cu
            src/triangles_counting_kernels.cu
            src/triangles_counting.cpp
            src/pagerank.cu
            src/pagerank_kernels.cu
            src/partition.cu
            src/valued_csr_graph.cpp
            src/widest_path.cu         
)

target_compile_definitions(nvgraph PRIVATE
    __HIP_PLATFORM_AMD__=1 
)

target_compile_options(nvgraph PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-std=c++14> 
)

###################################################################################################
# - build options ---------------------------------------------------------------------------------

if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Building with debugging flags")
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -g -O0")
endif(CMAKE_BUILD_TYPE MATCHES Debug)

if(NVGRAPH_LIGHT MATCHES True)
    add_definitions( -DNVGRAPH_LIGHT=${NVGRAPH_LIGHT} )
endif(NVGRAPH_LIGHT MATCHES True)


###################################################################################################
# - link libraries --------------------------------------------------------------------------------

target_link_libraries(nvgraph 
    PRIVATE
    -Wl,--start-group 
    hipblas hipsparse hiprand hipsolver amdhip64 hip::host cblas blas lapack
    -Wl,--end-group
)
# target_link_libraries(nvgraph 
#     PRIVATE
#         hipblas 
#         rocsparse
#         rocrand 
#         rocsolver 
#         amdhip64
#         hipsparse
#         hip::host
# )
# hip_hcc amdhip64 stdc++ gcc_s gcc

###################################################################################################
# - install targets -------------------------------------------------------------------------------

install(TARGETS nvgraph
        DESTINATION lib64)

install(FILES
         ${CMAKE_CURRENT_SOURCE_DIR}/include/nvgraph.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/test_opt_utils.cuh
        DESTINATION include/nvgraph)
