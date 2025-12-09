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

#if !defined(CUSPARSE_INTERNAL_H_)
#define CUSPARSE_INTERNAL_H_

#include <hip/hip_fp16.h>

#ifndef CUSPARSEAPI
#ifdef _WIN32
#define CUSPARSEAPI __stdcall
#else
#define CUSPARSEAPI 
#endif
#endif

#define CACHE_LINE_SIZE   128 

#define ALIGN_32(x)   ((((x)+31)/32)*32)



#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */


struct csrilu02BatchInfo;
typedef struct csrilu02BatchInfo *csrilu02BatchInfo_t;


struct csrxilu0Info;
typedef struct csrxilu0Info *csrxilu0Info_t;

struct csrxgemmSchurInfo;
typedef struct csrxgemmSchurInfo *csrxgemmSchurInfo_t;

struct csrxtrsmInfo;
typedef struct csrxtrsmInfo  *csrxtrsmInfo_t;

struct csrilu03Info;
typedef struct csrilu03Info *csrilu03Info_t;

struct csrmmInfo;
typedef struct csrmmInfo *csrmmInfo_t;


hipStream_t cusparseGetStreamInternal(const struct cusparseContext *ctx);


hipsparseStatus_t CUSPARSEAPI cusparseCheckBuffer(
    hipsparseHandle_t handle,
    void *workspace);

//------- gather: dst = src(map) ---------------------

hipsparseStatus_t CUSPARSEAPI cusparseIgather(
    hipsparseHandle_t handle,
    int n,
    const int *src,
    const int *map,
    int *dst);

hipsparseStatus_t CUSPARSEAPI cusparseSgather(
    hipsparseHandle_t handle,
    int n,
    const float *src,
    const int *map,
    float *dst);

hipsparseStatus_t CUSPARSEAPI cusparseDgather(
    hipsparseHandle_t handle,
    int n,
    const double *src,
    const int *map,
    double *dst);

hipsparseStatus_t CUSPARSEAPI cusparseCgather(
    hipsparseHandle_t handle,
    int n,
    const hipComplex *src,
    const int *map,
    hipComplex *dst);

hipsparseStatus_t CUSPARSEAPI cusparseZgather(
    hipsparseHandle_t handle,
    int n,
    const hipDoubleComplex *src,
    const int *map,
    hipDoubleComplex *dst);


//------- scatter: dst(map) = src ---------------------

hipsparseStatus_t CUSPARSEAPI cusparseIscatter(
    hipsparseHandle_t handle,
    int n,
    const int *src,
    int *dst,
    const int *map);

hipsparseStatus_t CUSPARSEAPI cusparseSscatter(
    hipsparseHandle_t handle,
    int n,
    const float *src,
    float *dst,
    const int *map);

hipsparseStatus_t CUSPARSEAPI cusparseDscatter(
    hipsparseHandle_t handle,
    int n,
    const double *src,
    double *dst,
    const int *map);

hipsparseStatus_t CUSPARSEAPI cusparseCscatter(
    hipsparseHandle_t handle,
    int n,
    const hipComplex *src,
    hipComplex *dst,
    const int *map);

hipsparseStatus_t CUSPARSEAPI cusparseZscatter(
    hipsparseHandle_t handle,
    int n,
    const hipDoubleComplex *src,
    hipDoubleComplex *dst,
    const int *map);


// x[j] = j 
hipsparseStatus_t CUSPARSEAPI cusparseIidentity(
    hipsparseHandle_t handle,
    int n,
    int *x);

// x[j] = val
hipsparseStatus_t CUSPARSEAPI cusparseImemset(
    hipsparseHandle_t handle,
    int n,
    int val,
    int *x);

hipsparseStatus_t CUSPARSEAPI cusparseI64memset(
    hipsparseHandle_t handle,
    size_t n,
    int val,
    int *x);


// ----------- reduce -----------------

/*
 * hipsparseStatus_t 
 *      cusparseIreduce_bufferSize( hipsparseHandle_t handle,
 *                                   int n,
 *                                   int *pBufferSizeInBytes)
 * Input
 * -----
 * handle        handle to CUSPARSE library context.
 * n             number of elements.
 *
 * Output
 * ------
 * pBufferSizeInBytes   size of working space in bytes.
 *  
 * Error Status
 * ------------
 * HIPSPARSE_STATUS_SUCCESS          the operation completed successfully.
 * HIPSPARSE_STATUS_NOT_INITIALIZED  the library was not initialized.   
 * HIPSPARSE_STATUS_INVALID_VALUE    n is too big or negative
 * HIPSPARSE_STATUS_INTERNAL_ERROR   an internal operation failed.
 *                                  If n is normal, we should not have this internal error.
 *
 * ---------
 * Assumption:
 *    Only support n < 2^31.
 *
 */
hipsparseStatus_t CUSPARSEAPI cusparseIreduce_bufferSizeExt(
    hipsparseHandle_t handle,
    int n,
    size_t *pBufferSizeInBytes);

/*
 * hipsparseStatus_t 
 *     cusparseIreduce(hipsparseHandle_t handle,
 *                     int n,
 *                     int *src,
 *                     int *pBuffer,
 *                     int *total_sum)
 *  
 *    total_sum = reduction(src)
 *
 *  Input
 * -------
 *  handle            handle to the CUSPARSE library context.
 *    n               number of elements in src and dst.
 *  src               <int> array of n elements.
 *  pBuffer           working space, the size is reported by cusparseIinclusiveScan_bufferSizeExt.
 *                    Or it can be a NULL pointer, then CUSPARSE library allocates working space implicitly.
 *
 * Output
 * -------
 *  total_sum         total_sum = reduction(src) if total_sum is not a NULL pointer.
 *
 *
 * Error Status
 * ------------
 * HIPSPARSE_STATUS_SUCCESS          the operation completed successfully.
 * HIPSPARSE_STATUS_NOT_INITIALIZED  the library was not initialized.   
 * HIPSPARSE_STATUS_ALLOC_FAILED     the resources could not be allocated.
 *                                  it is possible if pBuffer is NULL.
 * HIPSPARSE_STATUS_INTERNAL_ERROR   an internal operation failed.
 *
 * 
 */
hipsparseStatus_t CUSPARSEAPI cusparseIreduce(
    hipsparseHandle_t handle,
    int n,
    int *src,
    void *pBuffer,
    int *total_sum);



// ----------- prefix sum -------------------

/*
 * hipsparseStatus_t 
 *      cusparseIinclusiveScan_bufferSizeExt( hipsparseHandle_t handle,
 *                                   int n,
 *                                   size_t *pBufferSizeInBytes)
 * Input
 * -----
 * handle        handle to CUSPARSE library context.
 * n             number of elements.
 *
 * Output
 * ------
 * pBufferSizeInBytes   size of working space in bytes.
 *  
 * Error Status
 * ------------
 * HIPSPARSE_STATUS_SUCCESS          the operation completed successfully.
 * HIPSPARSE_STATUS_NOT_INITIALIZED  the library was not initialized.   
 * HIPSPARSE_STATUS_INVALID_VALUE    n is too big or negative
 * HIPSPARSE_STATUS_INTERNAL_ERROR   an internal operation failed.
 *                                  If n is normal, we should not have this internal error.
 *
 * ---------
 * Assumption:
 *    Only support n < 2^31.
 *
 */
hipsparseStatus_t CUSPARSEAPI cusparseIinclusiveScan_bufferSizeExt(
    hipsparseHandle_t handle,
    int n,
    size_t *pBufferSizeInBytes);


/*
 * hipsparseStatus_t 
 *     cusparseIinclusiveScan(hipsparseHandle_t handle,
 *                             int base,
 *                             int n,
 *                             int *src,
 *                             void *pBuffer,
 *                             int *dst,
 *                             int *total_sum)
 *  
 *    dst = inclusiveScan(src) + base
 *    total_sum = reduction(src)
 *
 *  Input
 * -------
 *  handle            handle to the CUSPARSE library context.
 *    n               number of elements in src and dst.
 *  src               <int> array of n elements.
 *  pBuffer           working space, the size is reported by cusparseIinclusiveScan_bufferSizeExt.
 *                    Or it can be a NULL pointer, then CUSPARSE library allocates working space implicitly.
 *
 * Output
 * -------
 *  dst               <int> array of n elements.
 *                    dst = inclusiveScan(src) + base
 *  total_sum         total_sum = reduction(src) if total_sum is not a NULL pointer.
 *
 * Error Status
 * ------------
 * HIPSPARSE_STATUS_SUCCESS          the operation completed successfully.
 * HIPSPARSE_STATUS_NOT_INITIALIZED  the library was not initialized.   
 * HIPSPARSE_STATUS_ALLOC_FAILED     the resources could not be allocated.
 *                                  it is possible if pBuffer is NULL.
 * HIPSPARSE_STATUS_INTERNAL_ERROR   an internal operation failed.
 * 
 */
hipsparseStatus_t CUSPARSEAPI cusparseIinclusiveScan(
    hipsparseHandle_t handle,
    int base,
    int n,
    int *src,
    void *pBuffer,
    int *dst,
    int *total_sum);

// ----------- stable sort -----------------

/*
 * hipsparseStatus_t 
 *      cusparseIstableSortByKey_bufferSizeExt( hipsparseHandle_t handle,
 *                                   int n,
 *                                   size_t *pBufferSizeInBytes)
 * Input
 * -----
 * handle        handle to CUSPARSE library context.
 * n             number of elements.
 *
 * Output
 * ------
 * pBufferSizeInBytes   size of working space in bytes.
 *  
 * Error Status
 * ------------
 * HIPSPARSE_STATUS_SUCCESS          the operation completed successfully.
 * HIPSPARSE_STATUS_NOT_INITIALIZED  the library was not initialized.   
 * HIPSPARSE_STATUS_INVALID_VALUE    n is too big or negative
 * HIPSPARSE_STATUS_INTERNAL_ERROR   an internal operation failed.
 *                                  If n is normal, we should not have this internal error.
 *
 * ---------
 * Assumption:
 *    Only support n < 2^30 because of domino scheme. 
 *
 */
hipsparseStatus_t CUSPARSEAPI cusparseIstableSortByKey_bufferSizeExt(
    hipsparseHandle_t handle,
    int n,
    size_t *pBufferSizeInBytes);


/*
 * hipsparseStatus_t 
 *      cusparseIstableSortByKey( hipsparseHandle_t handle,
 *                                   int n,
 *                                   int *key,
 *                                   int *P)
 *
 *  in-place radix sort. 
 *  This is an inhouse design of thrust::stable_sort_by_key(key, P)
 *
 * Input
 * -----
 * handle    handle to CUSPARSE library context.
 * n         number of elements.
 * key       <int> array of n elements.  
 * P         <int> array of n elements.  
 * pBuffer   working space, the size is reported by cusparseIstableSortByKey_bufferSize.
 *           Or it can be a NULL pointer, then CUSPARSE library allocates working space implicitly.
 *
 * Output
 * ------
 * key       <int> array of n elements.  
 * P         <int> array of n elements.  
 *
 * Error Status
 * ------------
 * HIPSPARSE_STATUS_SUCCESS          the operation completed successfully.
 * HIPSPARSE_STATUS_NOT_INITIALIZED  the library was not initialized.   
 * HIPSPARSE_STATUS_ALLOC_FAILED     the resources could not be allocated.
 * HIPSPARSE_STATUS_INTERNAL_ERROR   an internal operation failed.
 *
 * -----
 * Assumption:
 *    Only support n < 2^30 because of domino scheme. 
 *
 * -----
 * Usage:
 *   int nBufferSize = 0;
 *   status = cusparseIstableSortByKey_bufferSize(handle, n, &nBufferSize);
 *   assert(HIPSPARSE_STATUS_SUCCESS == status);
 *   
 *   int *pBuffer;
 *   cudaStat = hipMalloc((void**)&pBuffer, (size_t)nBufferSize);
 *   assert(hipSuccess == cudaStat);
 *
 *   d_P = 0:n-1 ;
 *   status = cusparseIstableSortByKey(handle, n, d_csrRowPtrA, d_P, pBuffer);
 *   assert(HIPSPARSE_STATUS_SUCCESS == status);
 *
 */
hipsparseStatus_t CUSPARSEAPI cusparseIstableSortByKey(
    hipsparseHandle_t handle,
    int n,
    int *key,
    int *P,
    void *pBuffer);



// ------------------- csr42csr ------------------

hipsparseStatus_t CUSPARSEAPI cusparseXcsr42csr_bufferSize(
    hipsparseHandle_t handle,
    int m,
    int n,
    const hipsparseMatDescr_t descrA,
    int nnzA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    size_t *pBufferSizeInByte );

hipsparseStatus_t CUSPARSEAPI cusparseXcsr42csrRows(
    hipsparseHandle_t handle,
    int m,
    int n,
    const hipsparseMatDescr_t descrA,
    int nnzA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    const hipsparseMatDescr_t descrC,
    int *csrRowPtrC,
    int *nnzTotalDevHostPtr,
    void *pBuffer );

hipsparseStatus_t CUSPARSEAPI cusparseXcsr42csrCols(
    hipsparseHandle_t handle,
    int m,
    int n,
    const hipsparseMatDescr_t descrA,
    int nnzA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    const hipsparseMatDescr_t descrC,
    const int *csrRowPtrC,
    int *csrColIndC,
    void *pBuffer );

hipsparseStatus_t CUSPARSEAPI cusparseScsr42csrVals(
    hipsparseHandle_t handle,
    int m,
    int n,
    const float *alpha,
    const hipsparseMatDescr_t descrA,
    int nnzA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    const hipsparseMatDescr_t descrC,
    float *csrValC,
    const int *csrRowPtrC,
    const int *csrEndPtrC,
    const int *csrColIndC,
    void *pBuffer );

hipsparseStatus_t CUSPARSEAPI cusparseDcsr42csrVals(
    hipsparseHandle_t handle,
    int m,
    int n,
    const double *alpha,
    const hipsparseMatDescr_t descrA,
    int nnzA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    const hipsparseMatDescr_t descrC,
    double *csrValC,
    const int *csrRowPtrC,
    const int *csrEndPtrC,
    const int *csrColIndC,
    void *pBuffer );

hipsparseStatus_t CUSPARSEAPI cusparseCcsr42csrVals(
    hipsparseHandle_t handle,
    int m,
    int n,
    const hipComplex *alpha,
    const hipsparseMatDescr_t descrA,
    int nnzA,
    const hipComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    const hipsparseMatDescr_t descrC,
    hipComplex *csrValC,
    const int *csrRowPtrC,
    const int *csrEndPtrC,
    const int *csrColIndC,
    void *pBuffer );

hipsparseStatus_t CUSPARSEAPI cusparseZcsr42csrVals(
    hipsparseHandle_t handle,
    int m,
    int n,
    const hipDoubleComplex *alpha,
    const hipsparseMatDescr_t descrA,
    int nnzA,
    const hipDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    const hipsparseMatDescr_t descrC,
    hipDoubleComplex *csrValC,
    const int *csrRowPtrC,
    const int *csrEndPtrC,
    const int *csrColIndC,
    void *pBuffer );


// ----- csrmv_hyb ------------------------------

hipsparseStatus_t CUSPARSEAPI cusparseScsrmv_hyb(
    hipsparseHandle_t handle,
    hipsparseOperation_t trans,
    int m,
    int n,
    int nnz,
    const float *alpha,
    const hipsparseMatDescr_t descra,
    const float *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const float *x,
    const float *beta,
    float *y);

hipsparseStatus_t CUSPARSEAPI cusparseDcsrmv_hyb(
    hipsparseHandle_t handle,
    hipsparseOperation_t trans,
    int m,
    int n,
    int nnz,
    const double *alpha,
    const hipsparseMatDescr_t descra,
    const double *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const double *x,
    const double *beta, 
    double *y);

hipsparseStatus_t CUSPARSEAPI cusparseCcsrmv_hyb(
    hipsparseHandle_t handle,
    hipsparseOperation_t trans,
    int m,
    int n,
    int nnz,
    const hipComplex *alpha,
    const hipsparseMatDescr_t descra,
    const hipComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const hipComplex *x,
    const hipComplex *beta,
    hipComplex *y);

hipsparseStatus_t CUSPARSEAPI cusparseZcsrmv_hyb(
    hipsparseHandle_t handle,
    hipsparseOperation_t trans,
    int m,
    int n,
    int nnz,
    const hipDoubleComplex *alpha,
    const hipsparseMatDescr_t descra,
    const hipDoubleComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const hipDoubleComplex *x,
    const hipDoubleComplex *beta,
    hipDoubleComplex *y);


// ------------- getrf_ilu ---------------------

hipsparseStatus_t CUSPARSEAPI cusparseSgetrf_ilu(
    hipsparseHandle_t handle,
    const int submatrix_k,
    const int n,
    float *A,
    const int *pattern,
    const int lda,
    int *d_status,
    int enable_boost,
    double *tol_ptr,
    float *boost_ptr);

hipsparseStatus_t CUSPARSEAPI cusparseDgetrf_ilu(
    hipsparseHandle_t handle,
    const int submatrix_k,
    const int n,
    double *A,
    const int *pattern,
    const int lda,
    int *d_status,
    int enable_boost,
    double *tol_ptr,
    double *boost_ptr);

hipsparseStatus_t CUSPARSEAPI cusparseCgetrf_ilu(
    hipsparseHandle_t handle,
    const int submatrix_k,
    const int n,
    hipComplex *A,
    const int *pattern,
    const int lda,
    int *d_status,
    int enable_boost,
    double *tol_ptr,
    hipComplex *boost_ptr);

hipsparseStatus_t CUSPARSEAPI cusparseZgetrf_ilu(
    hipsparseHandle_t handle,
    const int submatrix_k,
    const int n,
    hipDoubleComplex *A,
    const int *pattern,
    const int lda,
    int *d_status,
    int enable_boost,
    double *tol_ptr,
    hipDoubleComplex *boost_ptr);


// ------------- potrf_ic ---------------------

hipsparseStatus_t CUSPARSEAPI cusparseSpotrf_ic(
    hipsparseHandle_t handle,
    const int submatrix_k,
    const int n,
    float *A,
    const int *pattern,
    const int lda,
    int *d_status);

hipsparseStatus_t CUSPARSEAPI cusparseDpotrf_ic(
    hipsparseHandle_t handle,
    const int submatrix_k,
    const int n,
    double *A,
    const int *pattern,
    const int lda,
    int *d_status);

hipsparseStatus_t CUSPARSEAPI cusparseCpotrf_ic(
    hipsparseHandle_t handle,
    const int submatrix_k,
    const int n,
    hipComplex *A,
    const int *pattern,
    const int lda,
    int *d_status);

hipsparseStatus_t CUSPARSEAPI cusparseZpotrf_ic(
    hipsparseHandle_t handle,
    const int submatrix_k,
    const int n,
    hipDoubleComplex *A,
    const int *pattern,
    const int lda,
    int *d_status);


hipsparseStatus_t CUSPARSEAPI cusparseXcsric02_denseConfig(
    csric02Info_t info,
    int enable_dense_block,
    int max_dim_dense_block,
    int threshold_dense_block,
    double ratio);

hipsparseStatus_t CUSPARSEAPI cusparseXcsric02_workspaceConfig(
    csric02Info_t info,
    int disable_workspace_limit);


hipsparseStatus_t CUSPARSEAPI cusparseXcsrilu02_denseConfig(
    csrilu02Info_t info,
    int enable_dense_block,
    int max_dim_dense_block,
    int threshold_dense_block,
    double ratio);

hipsparseStatus_t CUSPARSEAPI cusparseXcsrilu02_workspaceConfig(
    csrilu02Info_t info,
    int disable_workspace_limit);


hipsparseStatus_t CUSPARSEAPI cusparseXcsrilu02Batch_denseConfig(
    csrilu02BatchInfo_t info,
    int enable_dense_block,
    int max_dim_dense_block,
    int threshold_dense_block,
    double ratio);

hipsparseStatus_t CUSPARSEAPI cusparseXcsrilu02Batch_workspaceConfig(
    csrilu02BatchInfo_t info,
    int disable_workspace_limit);



// ---------------- csric02 internal ----------------
hipsparseStatus_t CUSPARSEAPI cusparseXcsric02_getLevel(
    csric02Info_t info,
    int **level_ref);

hipsparseStatus_t CUSPARSEAPI cusparseScsric02_internal(
    hipsparseHandle_t handle,
    int enable_potrf,
    int dense_block_start,
    //int dense_block_dim, // = m - dense_block_start
    int dense_block_lda,
    int *level,  // level is a permutation vector of 0:(m-1)
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    float *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    csric02Info_t info,
    hipsparseSolvePolicy_t policy,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseDcsric02_internal(
    hipsparseHandle_t handle,
    int enable_potrf,
    int dense_block_start,
    //int dense_block_dim, // = m - dense_block_start
    int dense_block_lda,
    int *level,  // level is a permutation vector of 0:(m-1)
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    double *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    csric02Info_t info,
    hipsparseSolvePolicy_t policy,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseCcsric02_internal(
    hipsparseHandle_t handle,
    int enable_potrf,
    int dense_block_start,
    //int dense_block_dim, // = m - dense_block_start
    int dense_block_lda,
    int *level,  // level is a permutation vector of 0:(m-1)
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    hipComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    csric02Info_t info,
    hipsparseSolvePolicy_t policy,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseZcsric02_internal(
    hipsparseHandle_t handle,
    int enable_potrf,
    int dense_block_start,
    //int dense_block_dim, // = m - dense_block_start
    int dense_block_lda,
    int *level,  // level is a permutation vector of 0:(m-1)
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    hipDoubleComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    csric02Info_t info,
    hipsparseSolvePolicy_t policy,
    void *pBuffer);

// csrilu02 internal

hipsparseStatus_t CUSPARSEAPI cusparseXcsrilu02_getLevel(
    csrilu02Info_t info,
    int **level_ref);

hipsparseStatus_t CUSPARSEAPI cusparseXcsrilu02_getCsrEndPtrL(
    csrilu02Info_t info,
    int **csrEndPtrL_ref);


// ----------------- batch ilu0 -----------------

hipsparseStatus_t CUSPARSEAPI cusparseCreateCsrilu02BatchInfo(
    csrilu02BatchInfo_t *info);

hipsparseStatus_t CUSPARSEAPI cusparseDestroyCsrilu02BatchInfo(
    csrilu02BatchInfo_t info);

hipsparseStatus_t CUSPARSEAPI cusparseXcsrilu02Batch_zeroPivot(
    hipsparseHandle_t handle,
    csrilu02BatchInfo_t info,
    int *position);

hipsparseStatus_t CUSPARSEAPI cusparseScsrilu02Batch_numericBoost(
    hipsparseHandle_t handle,
    csrilu02BatchInfo_t info,
    int enable_boost,
    double *tol,
    float *numeric_boost);

hipsparseStatus_t CUSPARSEAPI cusparseDcsrilu02Batch_numericBoost(
    hipsparseHandle_t handle,
    csrilu02BatchInfo_t info,
    int enable_boost,
    double *tol,
    double *numeric_boost);

hipsparseStatus_t CUSPARSEAPI cusparseCcsrilu02Batch_numericBoost(
    hipsparseHandle_t handle,
    csrilu02BatchInfo_t info,
    int enable_boost,
    double *tol,
    hipComplex *numeric_boost);

hipsparseStatus_t CUSPARSEAPI cusparseZcsrilu02Batch_numericBoost(
    hipsparseHandle_t handle,
    csrilu02BatchInfo_t info,
    int enable_boost,
    double *tol,
    hipDoubleComplex *numeric_boost);

hipsparseStatus_t CUSPARSEAPI cusparseScsrilu02Batch_bufferSizeExt(
    hipsparseHandle_t handle,
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    float *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrilu02BatchInfo_t info,
    size_t *pBufferSizeInBytes);

hipsparseStatus_t CUSPARSEAPI cusparseDcsrilu02Batch_bufferSizeExt(
    hipsparseHandle_t handle,
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    double *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrilu02BatchInfo_t info,
    size_t *pBufferSizeInBytes);

hipsparseStatus_t CUSPARSEAPI cusparseCcsrilu02Batch_bufferSizeExt(
    hipsparseHandle_t handle,
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    hipComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrilu02BatchInfo_t info,
    size_t *pBufferSizeInBytes);

hipsparseStatus_t CUSPARSEAPI cusparseZcsrilu02Batch_bufferSizeExt(
    hipsparseHandle_t handle,
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    hipDoubleComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrilu02BatchInfo_t info,
    size_t *pBufferSizeInBytes);


hipsparseStatus_t CUSPARSEAPI cusparseScsrilu02Batch_analysis(
    hipsparseHandle_t handle,
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    const float *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrilu02BatchInfo_t info,
    hipsparseSolvePolicy_t policy,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseDcsrilu02Batch_analysis(
    hipsparseHandle_t handle,
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    const double *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrilu02BatchInfo_t info,
    hipsparseSolvePolicy_t policy,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseCcsrilu02Batch_analysis(
    hipsparseHandle_t handle,
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    const hipComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrilu02BatchInfo_t info,
    hipsparseSolvePolicy_t policy,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseZcsrilu02Batch_analysis(
    hipsparseHandle_t handle,
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    const hipDoubleComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrilu02BatchInfo_t info,
    hipsparseSolvePolicy_t policy,
    void *pBuffer);


hipsparseStatus_t CUSPARSEAPI cusparseScsrilu02Batch(
    hipsparseHandle_t handle,
    int m,
    int nnz,
    const hipsparseMatDescr_t descra,
    float *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrilu02BatchInfo_t info,
    hipsparseSolvePolicy_t policy,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseDcsrilu02Batch(
    hipsparseHandle_t handle,
    int m,
    int nnz,
    const hipsparseMatDescr_t descra,
    double *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrilu02BatchInfo_t info,
    hipsparseSolvePolicy_t policy,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseCcsrilu02Batch(
    hipsparseHandle_t handle,
    int m,
    int nnz,
    const hipsparseMatDescr_t descra,
    hipComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrilu02BatchInfo_t info,
    hipsparseSolvePolicy_t policy,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseZcsrilu02Batch(
    hipsparseHandle_t handle,
    int m,
    int nnz,
    const hipsparseMatDescr_t descra,
    hipDoubleComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrilu02BatchInfo_t info,
    hipsparseSolvePolicy_t policy,
    void *pBuffer);

// --------------- csrsv2 batch --------------

hipsparseStatus_t CUSPARSEAPI cusparseScsrsv2Batch_bufferSizeExt(
    hipsparseHandle_t handle,
    hipsparseOperation_t transA,
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    float *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrsv2Info_t info,
    size_t *pBufferSizeInBytes);

hipsparseStatus_t CUSPARSEAPI cusparseDcsrsv2Batch_bufferSizeExt(
    hipsparseHandle_t handle,
    hipsparseOperation_t transA,
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    double *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrsv2Info_t info,
    size_t *pBufferSizeInBytes);

hipsparseStatus_t CUSPARSEAPI cusparseCcsrsv2Batch_bufferSizeExt(
    hipsparseHandle_t handle,
    hipsparseOperation_t transA,
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    hipComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrsv2Info_t info,
    size_t *pBufferSizeInBytes);

hipsparseStatus_t CUSPARSEAPI cusparseZcsrsv2Batch_bufferSizeExt(
    hipsparseHandle_t handle,
    hipsparseOperation_t transA,
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    hipDoubleComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrsv2Info_t info,
    size_t *pBufferSizeInBytes);

hipsparseStatus_t CUSPARSEAPI cusparseScsrsv2Batch_analysis(
    hipsparseHandle_t handle,
    hipsparseOperation_t transA,
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    const float *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrsv2Info_t info,
    hipsparseSolvePolicy_t policy,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseDcsrsv2Batch_analysis(
    hipsparseHandle_t handle,
    hipsparseOperation_t transA,
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    const double *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrsv2Info_t info,
    hipsparseSolvePolicy_t policy,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseCcsrsv2Batch_analysis(
    hipsparseHandle_t handle,
    hipsparseOperation_t transA,
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    const hipComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrsv2Info_t info,
    hipsparseSolvePolicy_t policy,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseZcsrsv2Batch_analysis(
    hipsparseHandle_t handle,
    hipsparseOperation_t transA,
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    const hipDoubleComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrsv2Info_t info,
    hipsparseSolvePolicy_t policy,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseXcsrsv2Batch_zeroPivot(
    hipsparseHandle_t handle,
    csrsv2Info_t info,
    int *position);


hipsparseStatus_t CUSPARSEAPI cusparseScsrsv2Batch_solve(
    hipsparseHandle_t handle,
    hipsparseOperation_t trans,
    int m,
    int nnz,
    const hipsparseMatDescr_t descra,
    const float *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    csrsv2Info_t info,
    const float *x,
    float *y,
    int batchSize,
    hipsparseSolvePolicy_t policy,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseDcsrsv2Batch_solve(
    hipsparseHandle_t handle,
    hipsparseOperation_t trans,
    int m,
    int nnz,
    const hipsparseMatDescr_t descra,
    const double *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    csrsv2Info_t info,
    const double *x,
    double *y,
    int batchSize,
    hipsparseSolvePolicy_t policy,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseCcsrsv2Batch_solve(
    hipsparseHandle_t handle,
    hipsparseOperation_t trans,
    int m,
    int nnz,
    const hipsparseMatDescr_t descra,
    const hipComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    csrsv2Info_t info,
    const hipComplex *x,
    hipComplex *y,
    int batchSize,
    hipsparseSolvePolicy_t policy,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseZcsrsv2Batch_solve(
    hipsparseHandle_t handle,
    hipsparseOperation_t trans,
    int m,
    int nnz,
    const hipsparseMatDescr_t descra,
    const hipDoubleComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    csrsv2Info_t info,
    const hipDoubleComplex *x,
    hipDoubleComplex *y,
    int batchSize,
    hipsparseSolvePolicy_t policy,
    void *pBuffer);

//-------------- csrgemm2 -------------

hipsparseStatus_t CUSPARSEAPI cusparseXcsrgemm2_spaceConfig(
    csrgemm2Info_t info,
    int disable_space_limit);

// internal-use only
hipsparseStatus_t CUSPARSEAPI cusparseXcsrgemm2Rows_bufferSize(
    hipsparseHandle_t handle,
    int m,
    int n,
    int k,

    const hipsparseMatDescr_t descrA,
    int nnzA,
    const int *csrRowPtrA,
    const int *csrColIndA,

    const hipsparseMatDescr_t descrB,
    int nnzB,
    const int *csrRowPtrB,
    const int *csrColIndB,

    csrgemm2Info_t info,
    size_t *pBufferSize );

// internal-use only
hipsparseStatus_t CUSPARSEAPI cusparseXcsrgemm2Cols_bufferSize(
    hipsparseHandle_t handle,
    int m,
    int n,
    int k,

    const hipsparseMatDescr_t descrA,
    int nnzA,
    const int *csrRowPtrA,
    const int *csrColIndA,

    const hipsparseMatDescr_t descrB,
    int nnzB,
    const int *csrRowPtrB,
    const int *csrColIndB,

    csrgemm2Info_t info,
    size_t *pBufferSize );



hipsparseStatus_t CUSPARSEAPI cusparseXcsrgemm2Rows(
    hipsparseHandle_t handle,
    int m,
    int n,
    int k,

    const hipsparseMatDescr_t descrA,
    int nnzA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    const hipsparseMatDescr_t descrB,
    int nnzB,
    const int *csrRowPtrB,
    const int *csrEndPtrB,
    const int *csrColIndB,

    const hipsparseMatDescr_t descrD,
    int nnzD,
    const int *csrRowPtrD,
    const int *csrEndPtrD,
    const int *csrColIndD,

    const hipsparseMatDescr_t descrC,
    int *csrRowPtrC,

    int *nnzTotalDevHostPtr,
    csrgemm2Info_t info,
    void *pBuffer );


hipsparseStatus_t CUSPARSEAPI cusparseXcsrgemm2Cols(
    hipsparseHandle_t handle,
    int m,
    int n,
    int k,

    const hipsparseMatDescr_t descrA,
    int nnzA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    const hipsparseMatDescr_t descrB,
    int nnzB,
    const int *csrRowPtrB,
    const int *csrEndPtrB,
    const int *csrColIndB,

    const hipsparseMatDescr_t descrD,
    int nnzD,
    const int *csrRowPtrD,
    const int *csrEndPtrD,
    const int *csrColIndD,

    const hipsparseMatDescr_t descrC,
    const int *csrRowPtrC,
    int *csrColIndC,

    csrgemm2Info_t info,
    void *pBuffer );

hipsparseStatus_t CUSPARSEAPI cusparseScsrgemm2Vals(
    hipsparseHandle_t handle,
    int m,
    int n,
    int k,

    const float *alpha,

    const hipsparseMatDescr_t descrA,
    int nnzA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    const hipsparseMatDescr_t descrB,
    int nnzB,
    const float *csrValB,
    const int *csrRowPtrB,
    const int *csrEndPtrB,
    const int *csrColIndB,

    const hipsparseMatDescr_t descrD,
    int nnzD,
    const float *csrValD,
    const int *csrRowPtrD,
    const int *csrEndPtrD,
    const int *csrColIndD,

    const float *beta,

    const hipsparseMatDescr_t descrC,
    float *csrValC,
    const int *csrRowPtrC,
    const int *csrEndPtrC,
    const int *csrColIndC,

    csrgemm2Info_t info,
    void *pBuffer );


hipsparseStatus_t CUSPARSEAPI cusparseDcsrgemm2Vals(
    hipsparseHandle_t handle,
    int m,
    int n,
    int k,

    const double *alpha,

    const hipsparseMatDescr_t descrA,
    int nnzA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    const hipsparseMatDescr_t descrB,
    int nnzB,
    const double *csrValB,
    const int *csrRowPtrB,
    const int *csrEndPtrB,
    const int *csrColIndB,

    const hipsparseMatDescr_t descrD,
    int nnzD,
    const double *csrValD,
    const int *csrRowPtrD,
    const int *csrEndPtrD,
    const int *csrColIndD,

    const double *beta,

    const hipsparseMatDescr_t descrC,
    double *csrValC,
    const int *csrRowPtrC,
    const int *csrEndPtrC,
    const int *csrColIndC,

    csrgemm2Info_t info,
    void *pBuffer );


hipsparseStatus_t CUSPARSEAPI cusparseCcsrgemm2Vals(
    hipsparseHandle_t handle,
    int m,
    int n,
    int k,

    const hipComplex *alpha,

    const hipsparseMatDescr_t descrA,
    int nnzA,
    const hipComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    const hipsparseMatDescr_t descrB,
    int nnzB,
    const hipComplex *csrValB,
    const int *csrRowPtrB,
    const int *csrEndPtrB,
    const int *csrColIndB,

    const hipsparseMatDescr_t descrD,
    int nnzD,
    const hipComplex *csrValD,
    const int *csrRowPtrD,
    const int *csrEndPtrD,
    const int *csrColIndD,

    const hipComplex *beta,

    const hipsparseMatDescr_t descrC,
    hipComplex *csrValC,
    const int *csrRowPtrC,
    const int *csrEndPtrC,
    const int *csrColIndC,

    csrgemm2Info_t info,
    void *pBuffer );


hipsparseStatus_t CUSPARSEAPI cusparseZcsrgemm2Vals(
    hipsparseHandle_t handle,
    int m,
    int n,
    int k,

    const hipDoubleComplex *alpha,

    const hipsparseMatDescr_t descrA,
    int nnzA,
    const hipDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    const hipsparseMatDescr_t descrB,
    int nnzB,
    const hipDoubleComplex *csrValB,
    const int *csrRowPtrB,
    const int *csrEndPtrB,
    const int *csrColIndB,

    const hipsparseMatDescr_t descrD,
    int nnzD,
    const hipDoubleComplex *csrValD,
    const int *csrRowPtrD,
    const int *csrEndPtrD,
    const int *csrColIndD,

    const hipDoubleComplex *beta,

    const hipsparseMatDescr_t descrC,
    hipDoubleComplex *csrValC,
    const int *csrRowPtrC,
    const int *csrEndPtrC,
    const int *csrColIndC,

    csrgemm2Info_t info,
    void *pBuffer );


// ---------------- csr2csc2

hipsparseStatus_t CUSPARSEAPI cusparseXcsr2csc2_bufferSizeExt(
    hipsparseHandle_t handle,
    int m,
    int n,
    int nnz,
    const int *csrRowPtr,
    const int *csrColInd,
    size_t *pBufferSize);

hipsparseStatus_t CUSPARSEAPI cusparseXcsr2csc2(
    hipsparseHandle_t handle,
    int m,
    int n,
    int nnz,
    const hipsparseMatDescr_t descrA,
    const int *csrRowPtr,
    const int *csrColInd,
    int *cscColPtr,
    int *cscRowInd,
    int *cscValInd,
    void *pBuffer);

#if 0
// ------------- CSC ILU0

hipsparseStatus_t CUSPARSEAPI cusparseXcscilu02_getLevel(
    cscilu02Info_t info,
    int **level_ref);

hipsparseStatus_t CUSPARSEAPI cusparseXcscilu02_getCscColPtrL(
    cscilu02Info_t info,
    int **cscColPtrL_ref);

hipsparseStatus_t CUSPARSEAPI cusparseCreateCscilu02Info(
    cscilu02Info_t *info);

hipsparseStatus_t CUSPARSEAPI cusparseDestroyCscilu02Info(
    cscilu02Info_t info);

hipsparseStatus_t CUSPARSEAPI cusparseXcscilu02_zeroPivot(
    hipsparseHandle_t handle,
    cscilu02Info_t info,
    int *position);

hipsparseStatus_t CUSPARSEAPI cusparseScscilu02_numericBoost(
    hipsparseHandle_t handle,
    cscilu02Info_t info,
    int enable_boost,
    double *tol,
    float *numeric_boost);

hipsparseStatus_t CUSPARSEAPI cusparseDcscilu02_numericBoost(
    hipsparseHandle_t handle,
    cscilu02Info_t info,
    int enable_boost,
    double *tol,
    double *numeric_boost);

hipsparseStatus_t CUSPARSEAPI cusparseCcscilu02_numericBoost(
    hipsparseHandle_t handle,
    cscilu02Info_t info,
    int enable_boost,
    double *tol,
    hipComplex *numeric_boost);

hipsparseStatus_t CUSPARSEAPI cusparseZcscilu02_numericBoost(
    hipsparseHandle_t handle,
    cscilu02Info_t info,
    int enable_boost,
    double *tol,
    hipDoubleComplex *numeric_boost);

hipsparseStatus_t CUSPARSEAPI cusparseScscilu02_bufferSize(
    hipsparseHandle_t handle,
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    float *cscVal,
    const int *cscColPtr,
    const int *cscEndPtr,
    const int *cscRowInd,
    cscilu02Info_t info,
    int *pBufferSizeInBytes);

hipsparseStatus_t CUSPARSEAPI cusparseDcscilu02_bufferSize(
    hipsparseHandle_t handle,
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    double *cscVal,
    const int *cscColPtr,
    const int *cscEndPtr,
    const int *cscRowInd,
    cscilu02Info_t info,
    int *pBufferSizeInBytes);

hipsparseStatus_t CUSPARSEAPI cusparseCcscilu02_bufferSize(
    hipsparseHandle_t handle,
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    hipComplex *cscVal,
    const int *cscColPtr,
    const int *cscEndPtr,
    const int *cscRowInd,
    cscilu02Info_t info,
    int *pBufferSizeInBytes);

hipsparseStatus_t CUSPARSEAPI cusparseZcscilu02_bufferSize(
    hipsparseHandle_t handle,
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    hipDoubleComplex *cscVal,
    const int *cscColPtr,
    const int *cscEndPtr,
    const int *cscRowInd,
    cscilu02Info_t info,
    int *pBufferSizeInBytes);


hipsparseStatus_t CUSPARSEAPI cusparseScscilu02_analysis(
    hipsparseHandle_t handle,
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    const float *cscVal,
    const int *cscColPtr,
    const int *cscEndPtr,
    const int *cscRowInd,
    cscilu02Info_t info,
    hipsparseSolvePolicy_t policy,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseDcscilu02_analysis(
    hipsparseHandle_t handle,
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    const double *cscVal,
    const int *cscColPtr,
    const int *cscEndPtr,
    const int *cscRowInd,
    cscilu02Info_t info,
    hipsparseSolvePolicy_t policy,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseCcscilu02_analysis(
    hipsparseHandle_t handle,
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    const hipComplex *cscVal,
    const int *cscColPtr,
    const int *cscEndPtr,
    const int *cscRowInd,
    cscilu02Info_t info,
    hipsparseSolvePolicy_t policy,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseZcscilu02_analysis(
    hipsparseHandle_t handle,
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    const hipDoubleComplex *cscVal,
    const int *cscColPtr,
    const int *cscEndPtr,
    const int *cscRowInd,
    cscilu02Info_t info,
    hipsparseSolvePolicy_t policy,
    void *pBuffer);


hipsparseStatus_t CUSPARSEAPI cusparseScscilu02(
    hipsparseHandle_t handle,
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    float *cscVal,
    const int *cscColPtr,
    const int *cscEndPtr,
    const int *cscRowInd,
    cscilu02Info_t info,
    hipsparseSolvePolicy_t policy,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseDcscilu02(
    hipsparseHandle_t handle,
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    double *cscVal,
    const int *cscColPtr,
    const int *cscEndPtr,
    const int *cscRowInd,
    cscilu02Info_t info,
    hipsparseSolvePolicy_t policy,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseCcscilu02(
    hipsparseHandle_t handle,
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    hipComplex *cscVal,
    const int *cscColPtr,
    const int *cscEndPtr,
    const int *cscRowInd,
    cscilu02Info_t info,
    hipsparseSolvePolicy_t policy,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseZcscilu02(
    hipsparseHandle_t handle,
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    hipDoubleComplex *cscVal,
    const int *cscColPtr,
    const int *cscEndPtr,
    const int *cscRowInd,
    cscilu02Info_t info,
    hipsparseSolvePolicy_t policy,
    void *pBuffer);
#endif

// ------------- csrxjusqua

hipsparseStatus_t CUSPARSEAPI cusparseXcsrxjusqua(
    hipsparseHandle_t handle,
    int iax,
    int iay,
    int m,
    int n,
    const hipsparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,
    int *csrjusqua );

// ------------ csrxilu0

hipsparseStatus_t CUSPARSEAPI cusparseCreateCsrxilu0Info(
    csrxilu0Info_t *info);

hipsparseStatus_t CUSPARSEAPI cusparseDestroyCsrxilu0Info(
    csrxilu0Info_t info);

hipsparseStatus_t CUSPARSEAPI cusparseXcsrxilu0_zeroPivot(
    hipsparseHandle_t handle,
    csrxilu0Info_t info,
    int *position);

hipsparseStatus_t CUSPARSEAPI cusparseScsrxilu0_numericBoost(
    hipsparseHandle_t handle,
    csrxilu0Info_t info,
    int enable_boost,
    double *tol,
    float *numeric_boost);

hipsparseStatus_t CUSPARSEAPI cusparseDcsrxilu0_numericBoost(
    hipsparseHandle_t handle,
    csrxilu0Info_t info,
    int enable_boost,
    double *tol,
    double *numeric_boost);

hipsparseStatus_t CUSPARSEAPI cusparseCcsrxilu0_numericBoost(
    hipsparseHandle_t handle,
    csrxilu0Info_t info,
    int enable_boost,
    double *tol,
    hipComplex *numeric_boost);

hipsparseStatus_t CUSPARSEAPI cusparseZcsrxilu0_numericBoost(
    hipsparseHandle_t handle,
    csrxilu0Info_t info,
    int enable_boost,
    double *tol,
    hipDoubleComplex *numeric_boost);

hipsparseStatus_t CUSPARSEAPI cusparseXcsrxilu0_bufferSizeExt(
    hipsparseHandle_t handle,
    int iax,
    int iay,
    int m,
    int n,
    int k,
    const hipsparseMatDescr_t descrA,
    const int *csrRowPtr,
    const int *csrEndPtr,
    const int *csrColInd,
    csrxilu0Info_t info,
    size_t *pBufferSizeInBytes);

hipsparseStatus_t CUSPARSEAPI cusparseScsrxilu0(
    hipsparseHandle_t handle,
    int iax,
    int iay,
    int m,
    int n,
    int k,
    const hipsparseMatDescr_t descrA,
    float *csrVal,
    const int *csrRowPtr,
    const int *csrEndPtr,
    const int *csrColInd,
    csrxilu0Info_t info,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseDcsrxilu0(
    hipsparseHandle_t handle,
    int iax,
    int iay,
    int m,
    int n,
    int k,
    const hipsparseMatDescr_t descrA,
    double *csrVal,
    const int *csrRowPtr,
    const int *csrEndPtr,
    const int *csrColInd,
    csrxilu0Info_t info,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseCcsrxilu0(
    hipsparseHandle_t handle,
    int iax,
    int iay,
    int m,
    int n,
    int k,
    const hipsparseMatDescr_t descrA,
    hipComplex *csrVal,
    const int *csrRowPtr,
    const int *csrEndPtr,
    const int *csrColInd,
    csrxilu0Info_t info,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseZcsrxilu0(
    hipsparseHandle_t handle,
    int iax,
    int iay,
    int m,
    int n,
    int k,
    const hipsparseMatDescr_t descrA,
    hipDoubleComplex *csrVal,
    const int *csrRowPtr,
    const int *csrEndPtr,
    const int *csrColInd,
    csrxilu0Info_t info,
    void *pBuffer);

// ----------- csrxgemmSchur

hipsparseStatus_t CUSPARSEAPI cusparseCreateCsrxgemmSchurInfo(
    csrxgemmSchurInfo_t *info);

hipsparseStatus_t CUSPARSEAPI cusparseDestroyCsrxgemmSchurInfo(
    csrxgemmSchurInfo_t info);

hipsparseStatus_t CUSPARSEAPI cusparseXcsrxgemmSchur_bufferSizeExt(
    hipsparseHandle_t handle,
    int m,
    int n,
    int k,

    int iax,
    int iay,
    const hipsparseMatDescr_t descrA,
    int nnzA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    int ibx,
    int iby,
    const hipsparseMatDescr_t descrB,
    int nnzB,
    const int *csrRowPtrB,
    const int *csrEndPtrB,
    const int *csrColIndB,

    int icx,
    int icy,
    const hipsparseMatDescr_t descrC,
    int nnzC,
    const int *csrRowPtrC,
    const int *csrEndPtrC,
    const int *csrColIndC,

    csrxgemmSchurInfo_t info,
    size_t *pBufferSizeInBytes);


hipsparseStatus_t CUSPARSEAPI cusparseScsrxgemmSchur(
    hipsparseHandle_t handle,
    int m,
    int n,
    int k,

    int iax,
    int iay,
    const hipsparseMatDescr_t descrA,
    int nnzA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    int ibx,
    int iby,
    const hipsparseMatDescr_t descrB,
    int nnzB,
    const float *csrValB,
    const int *csrRowPtrB,
    const int *csrEndPtrB,
    const int *csrColIndB,

    int icx,
    int icy,
    const hipsparseMatDescr_t descrC,
    int nnzC,
    float *csrValC,
    const int *csrRowPtrC,
    const int *csrEndPtrC,
    const int *csrColIndC,

    csrxgemmSchurInfo_t info,
    void *pBuffer);


hipsparseStatus_t CUSPARSEAPI cusparseDcsrxgemmSchur(
    hipsparseHandle_t handle,
    int m,
    int n,
    int k,

    int iax,
    int iay,
    const hipsparseMatDescr_t descrA,
    int nnzA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    int ibx,
    int iby,
    const hipsparseMatDescr_t descrB,
    int nnzB,
    const double *csrValB,
    const int *csrRowPtrB,
    const int *csrEndPtrB,
    const int *csrColIndB,

    int icx,
    int icy,
    const hipsparseMatDescr_t descrC,
    int nnzC,
    double *csrValC,
    const int *csrRowPtrC,
    const int *csrEndPtrC,
    const int *csrColIndC,

    csrxgemmSchurInfo_t info,
    void *pBuffer);


hipsparseStatus_t CUSPARSEAPI cusparseCcsrxgemmSchur(
    hipsparseHandle_t handle,
    int m,
    int n,
    int k,

    int iax,
    int iay,
    const hipsparseMatDescr_t descrA,
    int nnzA,
    const hipComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    int ibx,
    int iby,
    const hipsparseMatDescr_t descrB,
    int nnzB,
    const hipComplex *csrValB,
    const int *csrRowPtrB,
    const int *csrEndPtrB,
    const int *csrColIndB,

    int icx,
    int icy,
    const hipsparseMatDescr_t descrC,
    int nnzC,
    hipComplex *csrValC,
    const int *csrRowPtrC,
    const int *csrEndPtrC,
    const int *csrColIndC,

    csrxgemmSchurInfo_t info,
    void *pBuffer);


hipsparseStatus_t CUSPARSEAPI cusparseZcsrxgemmSchur(
    hipsparseHandle_t handle,
    int m,
    int n,
    int k,

    int iax,
    int iay,
    const hipsparseMatDescr_t descrA,
    int nnzA,
    const hipDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    int ibx,
    int iby,
    const hipsparseMatDescr_t descrB,
    int nnzB,
    const hipDoubleComplex *csrValB,
    const int *csrRowPtrB,
    const int *csrEndPtrB,
    const int *csrColIndB,

    int icx,
    int icy,
    const hipsparseMatDescr_t descrC,
    int nnzC,
    hipDoubleComplex *csrValC,
    const int *csrRowPtrC,
    const int *csrEndPtrC,
    const int *csrColIndC,

    csrxgemmSchurInfo_t info,
    void *pBuffer);

// ---------- csrxtrsm

#if 0
hipsparseStatus_t CUSPARSEAPI cusparseCreateCsrxtrsmInfo(
    csrxtrsmInfo_t *info);

hipsparseStatus_t CUSPARSEAPI cusparseDestroyCsrxtrsmInfo(
    csrxtrsmInfo_t info);

hipsparseStatus_t CUSPARSEAPI cusparseXcsrxtrsm_bufferSizeExt(
    hipsparseHandle_t handle,
    int m,
    int n,

    cusparseSideMode_t side,

    int iax,
    int iay,
    const hipsparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    int ibx,
    int iby,
    const hipsparseMatDescr_t descrB,
    const int *csrRowPtrB,
    const int *csrEndPtrB,
    const int *csrColIndB,

    csrxtrsmInfo_t info,
    size_t *pBufferSizeInBytes);

hipsparseStatus_t  CUSPARSEAPI cusparseScsrxtrsm(
    hipsparseHandle_t handle,

    int m,
    int n,

    cusparseSideMode_t side,

    int iax,
    int iay,
    const hipsparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    int ibx,
    int iby,
    const hipsparseMatDescr_t descrB,
    float *csrValB,
    const int *csrRowPtrB,
    const int *csrEndPtrB,
    const int *csrColIndB,

    csrxtrsmInfo_t info,
    void *pBuffer);

hipsparseStatus_t  CUSPARSEAPI cusparseDcsrxtrsm(
    hipsparseHandle_t handle,

    int m,
    int n,

    cusparseSideMode_t side,

    int iax,
    int iay,
    const hipsparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    int ibx,
    int iby,
    const hipsparseMatDescr_t descrB,
    double *csrValB,
    const int *csrRowPtrB,
    const int *csrEndPtrB,
    const int *csrColIndB,

    csrxtrsmInfo_t info,
    void *pBuffer);

hipsparseStatus_t  CUSPARSEAPI cusparseCcsrxtrsm(
    hipsparseHandle_t handle,

    int m,
    int n,

    cusparseSideMode_t side,

    int iax,
    int iay,
    const hipsparseMatDescr_t descrA,
    const hipComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    int ibx,
    int iby,
    const hipsparseMatDescr_t descrB,
    hipComplex *csrValB,
    const int *csrRowPtrB,
    const int *csrEndPtrB,
    const int *csrColIndB,

    csrxtrsmInfo_t info,
    void *pBuffer);


hipsparseStatus_t  CUSPARSEAPI cusparseZcsrxtrsm(
    hipsparseHandle_t handle,

    int m,
    int n,

    cusparseSideMode_t side,

    int iax,
    int iay,
    const hipsparseMatDescr_t descrA,
    const hipDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    int ibx,
    int iby,
    const hipsparseMatDescr_t descrB,
    hipDoubleComplex *csrValB,
    const int *csrRowPtrB,
    const int *csrEndPtrB,
    const int *csrColIndB,

    csrxtrsmInfo_t info,
    void *pBuffer);
#endif

// ------ CSR ilu03
hipsparseStatus_t CUSPARSEAPI cusparseCreateCsrilu03Info(
    csrilu03Info_t *info);

hipsparseStatus_t CUSPARSEAPI cusparseDestroyCsrilu03Info(
    csrilu03Info_t info);

hipsparseStatus_t CUSPARSEAPI cusparseXcsrilu03_bufferSizeExt(
    hipsparseHandle_t handle,
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    const int *csrRowPtr,
    const int *csrColInd,
    csrilu03Info_t info,
    size_t *pBufferSizeInBytes);

hipsparseStatus_t CUSPARSEAPI cusparseXcsrilu03_zeroPivot(
    hipsparseHandle_t handle,
    csrilu03Info_t info,
    int *position);

hipsparseStatus_t CUSPARSEAPI cusparseScsrilu03_numericBoost(
    hipsparseHandle_t handle,
    csrilu03Info_t info,
    int enable_boost,
    double *tol,
    float *numeric_boost);

hipsparseStatus_t CUSPARSEAPI cusparseDcsrilu03_numericBoost(
    hipsparseHandle_t handle,
    csrilu03Info_t info,
    int enable_boost,
    double *tol,
    double *numeric_boost);

hipsparseStatus_t CUSPARSEAPI cusparseCcsrilu03_numericBoost(
    hipsparseHandle_t handle,
    csrilu03Info_t info,
    int enable_boost,
    double *tol,
    hipComplex *numeric_boost);

hipsparseStatus_t CUSPARSEAPI cusparseZcsrilu03_numericBoost(
    hipsparseHandle_t handle,
    csrilu03Info_t info,
    int enable_boost,
    double *tol,
    hipDoubleComplex *numeric_boost);

hipsparseStatus_t CUSPARSEAPI cusparseScsrilu03(
    hipsparseHandle_t handle,
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    float *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    csrilu03Info_t info,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseDcsrilu03(
    hipsparseHandle_t handle,
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    double *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    csrilu03Info_t info,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseCcsrilu03(
    hipsparseHandle_t handle,
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    hipComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    csrilu03Info_t info,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseZcsrilu03(
    hipsparseHandle_t handle,
    int m,
    int nnz,
    const hipsparseMatDescr_t descrA,
    hipDoubleComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    csrilu03Info_t info,
    void *pBuffer);


hipsparseStatus_t CUSPARSEAPI cusparseXcsrValid(
    hipsparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const hipsparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    int *valid);


hipsparseStatus_t CUSPARSEAPI cusparseScsrmm3(
    hipsparseHandle_t handle,
    hipsparseOperation_t transa,
    hipsparseOperation_t transb,
    int m,
    int n,
    int k,
    int nnz,
    const float *alpha,
    const hipsparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const float *B,
    int ldb,
    const float *beta,
    float *C,
    int ldc,
    void *buffer);

hipsparseStatus_t CUSPARSEAPI cusparseDcsrmm3(
    hipsparseHandle_t handle,
    hipsparseOperation_t transa,
    hipsparseOperation_t transb,
    int m,
    int n,
    int k,
    int nnz,
    const double *alpha,
    const hipsparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const double *B,
    int ldb,
    const double *beta,
    double *C,
    int ldc,
    void *buffer);

hipsparseStatus_t CUSPARSEAPI cusparseCcsrmm3(
    hipsparseHandle_t handle,
    hipsparseOperation_t transa,
    hipsparseOperation_t transb,
    int m,
    int n,
    int k,
    int nnz,
    const hipComplex *alpha,
    const hipsparseMatDescr_t descrA,
    const hipComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const hipComplex *B,
    int ldb,
    const hipComplex *beta,
    hipComplex *C,
    int ldc,
    void *buffer);

hipsparseStatus_t CUSPARSEAPI cusparseZcsrmm3(
    hipsparseHandle_t handle,
    hipsparseOperation_t transa,
    hipsparseOperation_t transb,
    int m,
    int n,
    int k,
    int nnz,
    const hipDoubleComplex *alpha,
    const hipsparseMatDescr_t descrA,
    const hipDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const hipDoubleComplex *B,
    int ldb,
    const hipDoubleComplex *beta,
    hipDoubleComplex *C,
    int ldc,
    void *buffer);

hipsparseStatus_t CUSPARSEAPI cusparseStranspose(
    hipsparseHandle_t handle,
    hipsparseOperation_t transa,
    int m,
    int n,
    const float *alpha,
    const float *A,
    int lda,
    float *C,
    int ldc);

hipsparseStatus_t CUSPARSEAPI cusparseDtranspose(
    hipsparseHandle_t handle,
    hipsparseOperation_t transa,
    int m,
    int n,
    const double *alpha,
    const double *A,
    int lda,
    double *C,
    int ldc);

hipsparseStatus_t CUSPARSEAPI cusparseCtranspose(
    hipsparseHandle_t handle,
    hipsparseOperation_t transa,
    int m,
    int n,
    const hipComplex *alpha,
    const hipComplex *A,
    int lda,
    hipComplex *C,
    int ldc);

hipsparseStatus_t CUSPARSEAPI cusparseZtranspose(
    hipsparseHandle_t handle,
    hipsparseOperation_t transa,
    int m,
    int n,
    const hipDoubleComplex *alpha,
    const hipDoubleComplex *A,
    int lda,
    hipDoubleComplex *C,
    int ldc);


hipsparseStatus_t CUSPARSEAPI cusparseScsrmv_binary(
    hipsparseHandle_t handle,
    hipsparseOperation_t trans,
    int m,
    int n,
    int nnz,
    const float *alpha,
    const hipsparseMatDescr_t descra,
    const int *csrRowPtr,
    const int *csrColInd,
    const float *x,
    const float *beta,
    float *y);

hipsparseStatus_t CUSPARSEAPI cusparseDcsrmv_binary(
    hipsparseHandle_t handle,
    hipsparseOperation_t trans,
    int m,
    int n,
    int nnz,
    const double *alpha,
    const hipsparseMatDescr_t descra,
    const int *csrRowPtr,
    const int *csrColInd,
    const double *x,
    const double *beta,
    double *y);

hipsparseStatus_t CUSPARSEAPI cusparseCcsrmv_binary(
    hipsparseHandle_t handle,
    hipsparseOperation_t trans,
    int m,
    int n,
    int nnz,
    const hipComplex *alpha,
    const hipsparseMatDescr_t descra,
    const int *csrRowPtr,
    const int *csrColInd,
    const hipComplex *x,
    const hipComplex *beta,
    hipComplex *y);

hipsparseStatus_t CUSPARSEAPI cusparseZcsrmv_binary(
    hipsparseHandle_t handle,
    hipsparseOperation_t trans,
    int m,
    int n,
    int nnz,
    const hipDoubleComplex *alpha,
    const hipsparseMatDescr_t descra,
    const int *csrRowPtr,
    const int *csrColInd,
    const hipDoubleComplex *x,
    const hipDoubleComplex *beta,
    hipDoubleComplex *y);

hipsparseStatus_t CUSPARSEAPI cusparseCreateCsrmmInfo(
    csrmmInfo_t *info);

hipsparseStatus_t CUSPARSEAPI cusparseDestroyCsrmmInfo(
    csrmmInfo_t info);

hipsparseStatus_t CUSPARSEAPI csrmm4_analysis(
    hipsparseHandle_t handle,
    int m, // number of rows of A
    int k, // number of columns of A
    int nnzA, // number of nonzeros of A
    const hipsparseMatDescr_t descrA,
    const int *csrRowPtrA, // <int> m+1
    const int *csrColIndA, // <int> nnzA
    csrmmInfo_t info,
    double *ratio // nnzB / nnzA
    );


hipsparseStatus_t CUSPARSEAPI cusparseScsrmm4(
    hipsparseHandle_t handle,
    hipsparseOperation_t transa,
    hipsparseOperation_t transb,
    int m,
    int n,
    int k,
    int nnz,
    const float *alpha,
    const hipsparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const float *B,
    int ldb,
    const float *beta,
    float *C,
    int ldc,
    csrmmInfo_t info,
    void *buffer);

hipsparseStatus_t CUSPARSEAPI cusparseDcsrmm4(
    hipsparseHandle_t handle,
    hipsparseOperation_t transa,
    hipsparseOperation_t transb,
    int m,
    int n,
    int k,
    int nnz,
    const double *alpha,
    const hipsparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const double *B,
    int ldb,
    const double *beta,
    double *C,
    int ldc,
    csrmmInfo_t info,
    void *buffer);

hipsparseStatus_t CUSPARSEAPI cusparseCcsrmm4(
    hipsparseHandle_t handle,
    hipsparseOperation_t transa,
    hipsparseOperation_t transb,
    int m,
    int n,
    int k,
    int nnz,
    const hipComplex *alpha,
    const hipsparseMatDescr_t descrA,
    const hipComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const hipComplex *B,
    int ldb,
    const hipComplex *beta,
    hipComplex *C,
    int ldc,
    csrmmInfo_t info,
    void *buffer);

hipsparseStatus_t CUSPARSEAPI cusparseZcsrmm4(
    hipsparseHandle_t handle,
    hipsparseOperation_t transa,
    hipsparseOperation_t transb,
    int m,
    int n,
    int k,
    int nnz,
    const hipDoubleComplex *alpha,
    const hipsparseMatDescr_t descrA,
    const hipDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const hipDoubleComplex *B,
    int ldb,
    const hipDoubleComplex *beta,
    hipDoubleComplex *C,
    int ldc,
    csrmmInfo_t info,
    void *buffer);

hipsparseStatus_t CUSPARSEAPI cusparseScsrmm5(
    hipsparseHandle_t handle,
    hipsparseOperation_t transa,
    hipsparseOperation_t transb,
    int m,
    int n,
    int k,
    int nnzA,
    const float *alpha,
    const hipsparseMatDescr_t descrA,
    const float  *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const float *B,
    int ldb,
    const float *beta,
    float *C,
    int ldc
    );

hipsparseStatus_t CUSPARSEAPI cusparseDcsrmm5(
    hipsparseHandle_t handle,
    hipsparseOperation_t transa,
    hipsparseOperation_t transb,
    int m,
    int n,
    int k,
    int nnzA,
    const double *alpha,
    const hipsparseMatDescr_t descrA,
    const double  *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const double *B,
    int ldb,
    const double *beta,
    double *C,
    int ldc
    );


hipsparseStatus_t CUSPARSEAPI cusparseScsrmm6(
    hipsparseHandle_t handle,
    hipsparseOperation_t transa,
    hipsparseOperation_t transb,
    int m,
    int n,
    int k,
    int nnzA,
    const float *alpha,
    const hipsparseMatDescr_t descrA,
    const float  *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const float *B,
    int ldb,
    const float *beta,
    float *C,
    int ldc
    );

hipsparseStatus_t CUSPARSEAPI cusparseDcsrmm6(
    hipsparseHandle_t handle,
    hipsparseOperation_t transa,
    hipsparseOperation_t transb,
    int m,
    int n,
    int k,
    int nnzA,
    const double *alpha,
    const hipsparseMatDescr_t descrA,
    const double  *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const double *B,
    int ldb,
    const double *beta,
    double *C,
    int ldc
    );



hipsparseStatus_t CUSPARSEAPI cusparseSmax(
    hipsparseHandle_t handle,
    int n,
    const float *x,
    float *valueHost,
    float *work  /* at least n+1 */
    );

hipsparseStatus_t CUSPARSEAPI cusparseDmax(
    hipsparseHandle_t handle,
    int n,
    const double *x,
    double *valueHost,
    double *work  /* at least n+1 */
    );

hipsparseStatus_t CUSPARSEAPI cusparseSmin(
    hipsparseHandle_t handle,
    int n,
    const float *x,
    float *valueHost,
    float *work  /* at least n+1 */
    );

hipsparseStatus_t CUSPARSEAPI cusparseDmin(
    hipsparseHandle_t handle,
    int n,
    const double *x,
    double *valueHost,
    double *work  /* at least n+1 */
    );

hipsparseStatus_t CUSPARSEAPI cusparseI16sort_internal_bufferSizeExt(
    hipsparseHandle_t handle,
    int n,
    size_t *pBufferSize);

hipsparseStatus_t CUSPARSEAPI cusparseI16sort_internal(
    hipsparseHandle_t handle,
    int num_bits, /* <= 16 */
    int n,
    unsigned short *key,
    int *P,
    int ascend,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseI32sort_internal_bufferSizeExt(
    hipsparseHandle_t handle,
    int n,
    size_t *pBufferSize);

hipsparseStatus_t CUSPARSEAPI cusparseI32sort_internal(
    hipsparseHandle_t handle,
    int num_bits, /* <= 32 */
    int n,
    unsigned int *key,
    int *P,
    int ascend,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseI64sort_internal_bufferSizeExt(
    hipsparseHandle_t handle,
    int n,
    size_t *pBufferSize);

hipsparseStatus_t CUSPARSEAPI cusparseI64sort_internal(
    hipsparseHandle_t handle,
    int num_bits, /* <= 64 */
    int n,
    unsigned long long *key,
    int *P,
    int ascend,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseIsort_bufferSizeExt(
    hipsparseHandle_t handle,
    int n,
    const int *key,
    const int *P,
    int ascend,
    size_t *pBufferSize);

hipsparseStatus_t CUSPARSEAPI cusparseIsort(
    hipsparseHandle_t handle,
    int n,
    int *key,
    int *P,
    int ascend,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseSsort_bufferSizeExt(
    hipsparseHandle_t handle,
    int n,
    const float *key,
    const int *P,
    int ascend,
    size_t *pBufferSize);

hipsparseStatus_t CUSPARSEAPI cusparseSsort(
    hipsparseHandle_t handle,
    int n,
    float *key,
    int *P,
    int ascend,
    void *pBuffer);


hipsparseStatus_t CUSPARSEAPI cusparseDsort_bufferSizeExt(
    hipsparseHandle_t handle,
    int n,
    const double *key,
    const int *P,
    int ascend,
    size_t *pBufferSize);

hipsparseStatus_t CUSPARSEAPI cusparseDsort(
    hipsparseHandle_t handle,
    int n,
    double *key,
    int *P,
    int ascend,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseHsort_bufferSizeExt(
    hipsparseHandle_t handle,
    int n,
    const __half *key,
    const int *P,
    int ascend,
    size_t *pBufferSize);

hipsparseStatus_t CUSPARSEAPI cusparseHsort(
    hipsparseHandle_t handle,
    int n,
    __half *key_fp16,
    int *P,
    int ascend,
    void *pBuffer);





hipsparseStatus_t CUSPARSEAPI cusparseHsortsign_bufferSizeExt(
    hipsparseHandle_t handle,
    int n,
    const __half *key,
    const int *P,
    int ascend,
    size_t *pBufferSize);

hipsparseStatus_t CUSPARSEAPI cusparseSsortsign_bufferSizeExt(
    hipsparseHandle_t handle,
    int n,
    const float *key,
    const int *P,
    int ascend,
    size_t *pBufferSize);

hipsparseStatus_t CUSPARSEAPI cusparseDsortsign_bufferSizeExt(
    hipsparseHandle_t handle,
    int n,
    const double *key,
    const int *P,
    int ascend,
    size_t *pBufferSize);

hipsparseStatus_t CUSPARSEAPI cusparseIsortsign_bufferSizeExt(
    hipsparseHandle_t handle,
    int n,
    const int *key,
    const int *P,
    int ascend,
    size_t *pBufferSize);

//#if defined(__cplusplus)
hipsparseStatus_t CUSPARSEAPI cusparseHsortsign(
    hipsparseHandle_t handle,
    int n,
    __half *key,
    int *P,
    int ascend,
    int *h_nnz_bucket0, /* host */
    void *pBuffer);
//#endif

hipsparseStatus_t CUSPARSEAPI cusparseSsortsign(
    hipsparseHandle_t handle,
    int n,
    float *key,
    int *P,
    int ascend,
    int *h_nnz_bucket0, /* host */
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseDsortsign(
    hipsparseHandle_t handle,
    int n,
    double *key,
    int *P,
    int ascend,
    int *h_nnz_bucket0, /* host */
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseIsortsign(
    hipsparseHandle_t handle,
    int n,
    int *key,
    int *P,
    int ascend,
    int *h_nnz_bucket0, /* host */
    void *pBuffer);

//----------------------------------------------


hipsparseStatus_t CUSPARSEAPI cusparseDDcsrMv_hyb(
    hipsparseHandle_t handle,
    hipsparseOperation_t trans,
    int m,
    int n,
    int nnz,
    const double *alpha,
    const hipsparseMatDescr_t descra,
    const double *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const double *x,
    const double *beta,
    double *y);


/*
 * gtsv2Batch: cuThomas algorithm
 * gtsv3Batch: QR
 * gtsv4Batch: LU with partial pivoting
 */
hipsparseStatus_t CUSPARSEAPI cusparseSgtsv2Batch(
    hipsparseHandle_t handle,
    int n,
    float *dl,
    float  *d,
    float *du,
    float *x,
    int batchCount);

hipsparseStatus_t CUSPARSEAPI cusparseDgtsv2Batch(
    hipsparseHandle_t handle,
    int n,
    double *dl,
    double  *d,
    double *du,
    double *x,
    int batchCount);

hipsparseStatus_t CUSPARSEAPI cusparseCgtsv2Batch(
    hipsparseHandle_t handle,
    int n,
    hipComplex *dl,
    hipComplex  *d,
    hipComplex *du,
    hipComplex *x,
    int batchCount);

hipsparseStatus_t CUSPARSEAPI cusparseZgtsv2Batch(
    hipsparseHandle_t handle,
    int n,
    hipDoubleComplex *dl,
    hipDoubleComplex  *d,
    hipDoubleComplex *du,
    hipDoubleComplex *x,
    int batchCount);

hipsparseStatus_t CUSPARSEAPI cusparseSgtsv3Batch_bufferSizeExt(
    hipsparseHandle_t handle,
    int n,
    const float *dl,
    const float  *d,
    const float *du,
    const float *x,
    int batchSize,
    size_t *pBufferSizeInBytes);

hipsparseStatus_t CUSPARSEAPI cusparseDgtsv3Batch_bufferSizeExt(
    hipsparseHandle_t handle,
    int n,
    const double *dl,
    const double  *d,
    const double *du,
    const double *x,
    int batchSize,
    size_t *pBufferSizeInBytes);

hipsparseStatus_t CUSPARSEAPI cusparseCgtsv3Batch_bufferSizeExt(
    hipsparseHandle_t handle,
    int n,
    const hipComplex *dl,
    const hipComplex  *d,
    const hipComplex *du,
    const hipComplex *x,
    int batchSize,
    size_t *pBufferSizeInBytes);

hipsparseStatus_t CUSPARSEAPI cusparseZgtsv3Batch_bufferSizeExt(
    hipsparseHandle_t handle,
    int n,
    const hipDoubleComplex *dl,
    const hipDoubleComplex  *d,
    const hipDoubleComplex *du,
    const hipDoubleComplex *x,
    int batchSize,
    size_t *pBufferSizeInBytes);

hipsparseStatus_t CUSPARSEAPI cusparseSgtsv3Batch(
    hipsparseHandle_t handle,
    int n,
    float *dl,
    float  *d,
    float *du,
    float *x,
    int batchSize,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseDgtsv3Batch(
    hipsparseHandle_t handle,
    int n,
    double *dl,
    double  *d,
    double *du,
    double *x,
    int batchSize,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseCgtsv3Batch(
    hipsparseHandle_t handle,
    int n,
    hipComplex *dl,
    hipComplex  *d,
    hipComplex *du,
    hipComplex *x,
    int batchSize,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseZgtsv3Batch(
    hipsparseHandle_t handle,
    int n,
    hipDoubleComplex *dl,
    hipDoubleComplex  *d,
    hipDoubleComplex *du,
    hipDoubleComplex *x,
    int batchSize,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseSgtsv4Batch_bufferSizeExt(
    hipsparseHandle_t handle,
    int n,
    const float *dl,
    const float  *d,
    const float *du,
    const float *x,
    int batchSize,
    size_t *pBufferSizeInBytes);

hipsparseStatus_t CUSPARSEAPI cusparseDgtsv4Batch_bufferSizeExt(
    hipsparseHandle_t handle,
    int n,
    const double *dl,
    const double  *d,
    const double *du,
    const double *x,
    int batchSize,
    size_t *pBufferSizeInBytes);

hipsparseStatus_t CUSPARSEAPI cusparseCgtsv4Batch_bufferSizeExt(
    hipsparseHandle_t handle,
    int n,
    const hipComplex *dl,
    const hipComplex  *d,
    const hipComplex *du,
    const hipComplex *x,
    int batchSize,
    size_t *pBufferSizeInBytes);

hipsparseStatus_t CUSPARSEAPI cusparseZgtsv4Batch_bufferSizeExt(
    hipsparseHandle_t handle,
    int n,
    const hipDoubleComplex *dl,
    const hipDoubleComplex  *d,
    const hipDoubleComplex *du,
    const hipDoubleComplex *x,
    int batchSize,
    size_t *pBufferSizeInBytes);

hipsparseStatus_t CUSPARSEAPI cusparseSgtsv4Batch(
    hipsparseHandle_t handle,
    int n,
    float *dl,
    float  *d,
    float *du,
    float *x,
    int batchSize,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseDgtsv4Batch(
    hipsparseHandle_t handle,
    int n,
    double *dl,
    double  *d,
    double *du,
    double *x,
    int batchSize,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseCgtsv4Batch(
    hipsparseHandle_t handle,
    int n,
    hipComplex *dl,
    hipComplex  *d,
    hipComplex *du,
    hipComplex *x,
    int batchSize,
    void *pBuffer);

hipsparseStatus_t CUSPARSEAPI cusparseZgtsv4Batch(
    hipsparseHandle_t handle,
    int n,
    hipDoubleComplex *dl,
    hipDoubleComplex  *d,
    hipDoubleComplex *du,
    hipDoubleComplex *x,
    int batchSize,
    void *pBuffer);


#if defined(__cplusplus)
}
#endif /* __cplusplus */


#endif /* CUSPARSE_INTERNAL_H_ */

