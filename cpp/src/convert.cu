/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <hipblas.h>
#include "nvgraph_convert.hxx"
#include "nvgraph_error.hxx"

namespace nvgraph
{
  void csr2coo(const int *csrSortedRowPtr,
               int nnz, int m, int *cooRowInd, hipsparseIndexBase_t idxBase)
  {
    CHECK_CUSPARSE(hipsparseXcsr2coo(Cusparse::get_handle(),
                                     csrSortedRowPtr, nnz, m, cooRowInd, idxBase));
  }
  void coo2csr(const int *cooRowInd,
               int nnz, int m, int *csrSortedRowPtr, hipsparseIndexBase_t idxBase)
  {
    CHECK_CUSPARSE(hipsparseXcoo2csr(Cusparse::get_handle(),
                                     cooRowInd, nnz, m, csrSortedRowPtr, idxBase));
  }

  void csr2csc(int m, int n, int nnz,
               const void *csrVal, const int *csrRowPtr, const int *csrColInd,
               void *cscVal, int *cscRowInd, int *cscColPtr,
               hipsparseAction_t copyValues, hipsparseIndexBase_t idxBase,
               hipblasDatatype_t *dataType)
  {
    // CHECK_CUSPARSE(cusparseCsr2cscEx(Cusparse::get_handle(),
    //                                    m, n, nnz,
    //                                    csrVal, *dataType, csrRowPtr, csrColInd,
    //                                    cscVal, *dataType, cscRowInd, cscColPtr,
    //                                    copyValues, idxBase, *dataType));

    hipsparseHandle_t handle = Cusparse::get_handle();
    hipDataType valType = static_cast<hipDataType>(*dataType);

    size_t buffer_size = 0;
    hipsparseCsr2CscAlg_t alg = HIPSPARSE_CSR2CSC_ALG1;

    CHECK_CUSPARSE(hipsparseCsr2cscEx2_bufferSize(
        handle,
        m, n, nnz,
        csrVal, csrRowPtr, csrColInd,
        cscVal, cscColPtr, cscRowInd,
        valType, copyValues, idxBase,
        alg,
        &buffer_size));

    void *buffer = nullptr;
    if (buffer_size > 0)
    {
      hipError_t hip_status = hipMalloc(&buffer, buffer_size);
      if (hip_status != hipSuccess)
      {
        CHECK_CUDA(hip_status);
      }
    }

    CHECK_CUSPARSE(hipsparseCsr2cscEx2(
        handle,
        m, n, nnz,
        csrVal, csrRowPtr, csrColInd,
        cscVal, cscColPtr, cscRowInd,
        valType,
        copyValues, idxBase,
        alg,
        buffer));

    if (buffer)
    {
      CHECK_CUDA(hipFree(buffer));
    }
  }

  void csc2csr(int m, int n, int nnz,
               const void *cscVal, const int *cscRowInd, const int *cscColPtr,
               void *csrVal, int *csrRowPtr, int *csrColInd,
               hipsparseAction_t copyValues, hipsparseIndexBase_t idxBase,
               hipblasDatatype_t *dataType)
  {
    // CHECK_CUSPARSE(cusparseCsr2cscEx(Cusparse::get_handle(),
    //                                    m, n, nnz,
    //                                    cscVal, *dataType, cscColPtr, cscRowInd,
    //                                    csrVal, *dataType, csrColInd, csrRowPtr,
    //                                    copyValues, idxBase, *dataType));

    hipsparseHandle_t handle = Cusparse::get_handle();
    hipDataType valType = static_cast<hipDataType>(*dataType);

    int m_csc = n;
    int n_csc = m;

    size_t buffer_size = 0;
    hipsparseCsr2CscAlg_t alg = HIPSPARSE_CSR2CSC_ALG1;

    CHECK_CUSPARSE(hipsparseCsr2cscEx2_bufferSize(
        handle,
        m_csc, n_csc, nnz,
        cscVal, cscColPtr, cscRowInd,
        csrVal, csrRowPtr, csrColInd,
        valType, copyValues, idxBase,
        alg,
        &buffer_size));

    void *buffer = nullptr;
    if (buffer_size > 0)
    {
      hipError_t hip_status = hipMalloc(&buffer, buffer_size);
      CHECK_CUDA(hip_status);
    }

    CHECK_CUSPARSE(hipsparseCsr2cscEx2(
        handle,
        m_csc, n_csc, nnz,
        cscVal, cscColPtr, cscRowInd,
        csrVal, csrRowPtr, csrColInd,
        valType,
        copyValues, idxBase,
        alg,
        buffer));

    if (buffer)
    {
      CHECK_CUDA(hipFree(buffer));
    }
  }

  void cooSortByDestination(int m, int n, int nnz,
                            const void *srcVal, const int *srcRowInd, const int *srcColInd,
                            void *dstVal, int *dstRowInd, int *dstColInd,
                            hipsparseIndexBase_t idxBase, hipblasDatatype_t *dataType)
  {
    size_t pBufferSizeInBytes = 0;
    SHARED_PREFIX::shared_ptr<char> pBuffer;
    SHARED_PREFIX::shared_ptr<int> P; // permutation array

    // step 0: copy src to dst
    if (dstRowInd != srcRowInd)
      CHECK_CUDA(hipMemcpy(dstRowInd, srcRowInd, nnz * sizeof(int), hipMemcpyDefault));
    if (dstColInd != srcColInd)
      CHECK_CUDA(hipMemcpy(dstColInd, srcColInd, nnz * sizeof(int), hipMemcpyDefault));
    // step 1: allocate buffer (needed for cooSortByRow)
    cooSortBufferSize(m, n, nnz, dstRowInd, dstColInd, &pBufferSizeInBytes);
    pBuffer = allocateDevice<char>(pBufferSizeInBytes, NULL);
    // step 2: setup permutation vector P to identity
    P = allocateDevice<int>(nnz, NULL);
    createIdentityPermutation(nnz, P.get());
    // step 3: sort COO format by Row
    cooGetDestinationPermutation(m, n, nnz, dstRowInd, dstColInd, P.get(), pBuffer.get());
    // step 4: gather sorted cooVals
    gthrX(nnz, srcVal, dstVal, P.get(), idxBase, dataType);
  }
  void cooSortBySource(int m, int n, int nnz,
                       const void *srcVal, const int *srcRowInd, const int *srcColInd,
                       void *dstVal, int *dstRowInd, int *dstColInd,
                       hipsparseIndexBase_t idxBase, hipblasDatatype_t *dataType)
  {
    size_t pBufferSizeInBytes = 0;
    SHARED_PREFIX::shared_ptr<char> pBuffer;
    SHARED_PREFIX::shared_ptr<int> P; // permutation array

    // step 0: copy src to dst
    CHECK_CUDA(hipMemcpy(dstRowInd, srcRowInd, nnz * sizeof(int), hipMemcpyDefault));
    CHECK_CUDA(hipMemcpy(dstColInd, srcColInd, nnz * sizeof(int), hipMemcpyDefault));
    // step 1: allocate buffer (needed for cooSortByRow)
    cooSortBufferSize(m, n, nnz, dstRowInd, dstColInd, &pBufferSizeInBytes);
    pBuffer = allocateDevice<char>(pBufferSizeInBytes, NULL);
    // step 2: setup permutation vector P to identity
    P = allocateDevice<int>(nnz, NULL);
    createIdentityPermutation(nnz, P.get());
    // step 3: sort COO format by Row
    cooGetSourcePermutation(m, n, nnz, dstRowInd, dstColInd, P.get(), pBuffer.get());
    // step 4: gather sorted cooVals
    gthrX(nnz, srcVal, dstVal, P.get(), idxBase, dataType);
  }

  void coos2csc(int m, int n, int nnz,
                const void *srcVal, const int *srcRowInd, const int *srcColInd,
                void *dstVal, int *dstRowInd, int *dstColPtr,
                hipsparseIndexBase_t idxBase, hipblasDatatype_t *dataType)
  {
    // coos -> cood -> csc
    SHARED_PREFIX::shared_ptr<int> tmp = allocateDevice<int>(nnz, NULL);
    cooSortByDestination(m, n, nnz, srcVal, srcRowInd, srcColInd, dstVal, dstRowInd, tmp.get(), idxBase, dataType);
    coo2csr(tmp.get(), nnz, m, dstColPtr, idxBase);
  }
  void cood2csr(int m, int n, int nnz,
                const void *srcVal, const int *srcRowInd, const int *srcColInd,
                void *dstVal, int *dstRowPtr, int *dstColInd,
                hipsparseIndexBase_t idxBase, hipblasDatatype_t *dataType)
  {
    // cood -> coos -> csr
    SHARED_PREFIX::shared_ptr<int> tmp = allocateDevice<int>(nnz, NULL);
    cooSortBySource(m, n, nnz, srcVal, srcRowInd, srcColInd, dstVal, tmp.get(), dstColInd, idxBase, dataType);
    coo2csr(tmp.get(), nnz, m, dstRowPtr, idxBase);
  }
  void coou2csr(int m, int n, int nnz,
                const void *srcVal, const int *srcRowInd, const int *srcColInd,
                void *dstVal, int *dstRowPtr, int *dstColInd,
                hipsparseIndexBase_t idxBase, hipblasDatatype_t *dataType)
  {
    cood2csr(m, n, nnz,
             srcVal, srcRowInd, srcColInd,
             dstVal, dstRowPtr, dstColInd,
             idxBase, dataType);
  }
  void coou2csc(int m, int n, int nnz,
                const void *srcVal, const int *srcRowInd, const int *srcColInd,
                void *dstVal, int *dstRowInd, int *dstColPtr,
                hipsparseIndexBase_t idxBase, hipblasDatatype_t *dataType)
  {
    coos2csc(m, n, nnz,
             srcVal, srcRowInd, srcColInd,
             dstVal, dstRowInd, dstColPtr,
             idxBase, dataType);
  }

  ////////////////////////// Utility functions //////////////////////////
  void createIdentityPermutation(int n, int *p)
  {
    CHECK_CUSPARSE(hipsparseCreateIdentityPermutation(Cusparse::get_handle(), n, p));
  }

  void gthrX(int nnz, const void *y, void *xVal, const int *xInd,
             hipsparseIndexBase_t idxBase, hipblasDatatype_t *dataType)
  {
    if (*dataType == HIPBLAS_R_32F)
    {
      CHECK_CUSPARSE(hipsparseSgthr(Cusparse::get_handle(), nnz, (float *)y, (float *)xVal, xInd, idxBase));
    }
    else if (*dataType == HIPBLAS_R_64F)
    {
      CHECK_CUSPARSE(hipsparseDgthr(Cusparse::get_handle(), nnz, (double *)y, (double *)xVal, xInd, idxBase));
    }
  }

  void cooSortBufferSize(int m, int n, int nnz, const int *cooRows, const int *cooCols, size_t *pBufferSizeInBytes)
  {
    CHECK_CUSPARSE(hipsparseXcoosort_bufferSizeExt(Cusparse::get_handle(),
                                                   m, n, nnz,
                                                   cooRows, cooCols, pBufferSizeInBytes));
  }
  void cooGetSourcePermutation(int m, int n, int nnz, int *cooRows, int *cooCols, int *p, void *pBuffer)
  {
    CHECK_CUSPARSE(hipsparseXcoosortByRow(Cusparse::get_handle(),
                                          m, n, nnz,
                                          cooRows, cooCols, p, pBuffer));
  }
  void cooGetDestinationPermutation(int m, int n, int nnz, int *cooRows, int *cooCols, int *p, void *pBuffer)
  {
    CHECK_CUSPARSE(hipsparseXcoosortByColumn(Cusparse::get_handle(),
                                             m, n, nnz,
                                             cooRows, cooCols, p, pBuffer));
  }
} // end namespace nvgraph
