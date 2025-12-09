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
 
#include <nvgraph_cusparse.hxx>

namespace nvgraph
{
hipsparseHandle_t Cusparse::m_handle = 0;

namespace
{
  hipsparseStatus_t cusparse_csrmv( hipsparseHandle_t handle, hipsparseOperation_t trans,
                              int m, int n, int nnz, 
                              const float *alpha, 
                              const hipsparseMatDescr_t descr,
                              const float *csrVal, 
                              const int *csrRowPtr, 
                              const int *csrColInd,
                              const float *x, 
                              const float *beta, 
                              float *y)
  {
      return hipsparseScsrmv(handle, trans, m, n, nnz, alpha, descr, csrVal, csrRowPtr, csrColInd, x, beta, y);
  }

  hipsparseStatus_t cusparse_csrmv( hipsparseHandle_t handle, hipsparseOperation_t trans,
                              int m, int n, int nnz, 
                              const double *alpha, 
                              const hipsparseMatDescr_t descr,
                              const double *csrVal, 
                              const int *csrRowPtr, 
                              const int *csrColInd,
                              const double *x, 
                              const double *beta, 
                              double *y)
  {
    return hipsparseDcsrmv(handle, trans, m, n, nnz, alpha, descr, csrVal, csrRowPtr, csrColInd, x, beta, y);
  }

  hipsparseStatus_t cusparse_csrmm(hipsparseHandle_t handle, hipsparseOperation_t trans,
                                  int m, int n, int k, int nnz, 
                                  const float *alpha, 
                                  const hipsparseMatDescr_t descr,
                                  const float *csrVal, 
                                  const int *csrRowPtr, 
                                  const int *csrColInd,
                                  const float *x,
                                  const int ldx,     
                                  const float *beta, 
                                  float *y,
                                  const int ldy)
  {
      return hipsparseScsrmm(handle, trans, m, n, k, nnz, alpha, descr, csrVal, csrRowPtr, csrColInd, x, ldx, beta, y, ldy);
  }

  hipsparseStatus_t cusparse_csrmm( hipsparseHandle_t handle, hipsparseOperation_t trans,
                                   int m, int n, int k, int nnz, 
                                   const double *alpha, 
                                   const hipsparseMatDescr_t descr,
                                   const double *csrVal, 
                                   const int *csrRowPtr, 
                                   const int *csrColInd,
                                   const double *x,
                                   const int ldx,
                                   const double *beta, 
                                   double *y,
                                   const int ldy)
  {
      return hipsparseDcsrmm(handle, trans, m, n, k, nnz, alpha, descr, csrVal, csrRowPtr, csrColInd, x, ldx, beta, y, ldy);
  }

}// end anonymous namespace.

// Set pointer mode
void Cusparse::set_pointer_mode_device()
{
    hipsparseHandle_t handle = Cusparse::get_handle();
    hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE);
}
void Cusparse::set_pointer_mode_host()
{
    hipsparseHandle_t handle = Cusparse::get_handle();
    hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST);
}

template <typename IndexType_, typename ValueType_>
void Cusparse::csrmv( const bool transposed,
                             const bool sym,
                             const int m, const int n, const int nnz, 
                             const ValueType_* alpha, 
                             const ValueType_* csrVal,
                             const IndexType_ *csrRowPtr, 
                             const IndexType_ *csrColInd, 
                             const ValueType_* x,
                             const ValueType_* beta, 
                             ValueType_* y)
{
  hipsparseHandle_t handle = Cusparse::get_handle();
  hipsparseOperation_t trans = transposed ? HIPSPARSE_OPERATION_TRANSPOSE : HIPSPARSE_OPERATION_NON_TRANSPOSE;
  hipsparseMatDescr_t descr=0;
  CHECK_CUSPARSE(hipsparseCreateMatDescr(&descr)); // we should move that somewhere else
  if (sym)
  {
    CHECK_CUSPARSE(hipsparseSetMatType(descr,HIPSPARSE_MATRIX_TYPE_SYMMETRIC));
  }
  else
  {
    CHECK_CUSPARSE(hipsparseSetMatType(descr,HIPSPARSE_MATRIX_TYPE_GENERAL));
  }
  CHECK_CUSPARSE(hipsparseSetMatIndexBase(descr,HIPSPARSE_INDEX_BASE_ZERO));
  CHECK_CUSPARSE(cusparse_csrmv(handle, trans , m, n, nnz, alpha, descr, csrVal, csrRowPtr, csrColInd, x, beta, y));
  CHECK_CUSPARSE(hipsparseDestroyMatDescr(descr)); // we should move that somewhere else
}

template <typename IndexType_, typename ValueType_>
void Cusparse::csrmv( const bool transposed,
                     const bool sym,
                     const ValueType_* alpha, 
                     const ValuedCsrGraph<IndexType_, ValueType_>& G,
                     const Vector<ValueType_>& x,
                     const ValueType_* beta, 
                     Vector<ValueType_>& y
                     )
{
  hipsparseHandle_t handle = Cusparse::get_handle();
  hipsparseOperation_t trans = transposed ? HIPSPARSE_OPERATION_TRANSPOSE : HIPSPARSE_OPERATION_NON_TRANSPOSE;
  hipsparseMatDescr_t descr=0;
  CHECK_CUSPARSE(hipsparseCreateMatDescr(&descr)); // we should move that somewhere else
  if (sym)
  {
    CHECK_CUSPARSE(hipsparseSetMatType(descr,HIPSPARSE_MATRIX_TYPE_SYMMETRIC));
  }
  else
  {
    CHECK_CUSPARSE(hipsparseSetMatType(descr,HIPSPARSE_MATRIX_TYPE_GENERAL));
  }
  int n = G.get_num_vertices();
  int nnz = G.get_num_edges();
  CHECK_CUSPARSE(hipsparseSetMatIndexBase(descr,HIPSPARSE_INDEX_BASE_ZERO));
  CHECK_CUSPARSE(cusparse_csrmv(handle, trans , n, n, nnz, alpha, descr, (ValueType_*)G.get_raw_values(), (IndexType_*)G.get_raw_row_offsets(),(IndexType_*)G.get_raw_column_indices(), (ValueType_*)x.raw(), beta,  (ValueType_*)y.raw()));
  CHECK_CUSPARSE(hipsparseDestroyMatDescr(descr)); // we should move that somewhere else
}

template void Cusparse::csrmv( const bool transposed,
                             const bool sym,
                             const int m, const int n, const int nnz, 
                             const double* alpha, 
                             const double* csrVal,
                             const int *csrRowPtr, 
                             const int *csrColInd, 
                             const double* x,
                             const double* beta, 
                             double* y);
template void Cusparse::csrmv( const bool transposed,
                             const bool sym,
                             const int m, const int n, const int nnz, 
                             const float* alpha, 
                             const float* csrVal,
                             const int *csrRowPtr, 
                             const int *csrColInd, 
                             const float* x,
                             const float* beta, 
                             float* y);
/*
template void Cusparse::csrmv( const bool transposed,
                               const bool sym,
                               const double* alpha, 
                               const ValuedCsrGraph<int, double>& G,
                               const Vector<double>& x,
                               const double* beta, 
                               Vector<double>& y
                     );


template void Cusparse::csrmv( const bool transposed,
                               const bool sym,
                               const float* alpha, 
                               const ValuedCsrGraph<int, float>& G,
                               const Vector<float>& x,
                               const float* beta, 
                               Vector<float>& y
                     );
*/


template <typename IndexType_, typename ValueType_>
void Cusparse::csrmm(const bool transposed,
                     const bool sym,
                     const int m, 
                     const int n, 
                     const int k,
                     const int nnz, 
                     const ValueType_* alpha, 
                     const ValueType_* csrVal,
                     const IndexType_* csrRowPtr, 
                     const IndexType_* csrColInd, 
                     const ValueType_* x,
                     const int ldx,
                     const ValueType_* beta, 
                     ValueType_* y,
                     const int ldy)
{

  hipsparseHandle_t handle = Cusparse::get_handle();
  hipsparseOperation_t trans = transposed ? HIPSPARSE_OPERATION_TRANSPOSE : HIPSPARSE_OPERATION_NON_TRANSPOSE;
  hipsparseMatDescr_t descr=0;
  CHECK_CUSPARSE(hipsparseCreateMatDescr(&descr)); // we should move that somewhere else
  if (sym)
  {
    CHECK_CUSPARSE(hipsparseSetMatType(descr,HIPSPARSE_MATRIX_TYPE_SYMMETRIC));
  }
  else
  {
    CHECK_CUSPARSE(hipsparseSetMatType(descr,HIPSPARSE_MATRIX_TYPE_GENERAL));
  }
  CHECK_CUSPARSE(hipsparseSetMatIndexBase(descr,HIPSPARSE_INDEX_BASE_ZERO));
  CHECK_CUSPARSE(cusparse_csrmm(handle, trans, m, n, k, nnz, alpha, descr, csrVal, csrRowPtr, csrColInd, x, ldx, beta, y, ldy));
  CHECK_CUSPARSE(hipsparseDestroyMatDescr(descr)); // we should move that somewhere else
}

template void Cusparse::csrmm(const bool transposed,
                              const bool sym,
                              const int m, 
                              const int n, 
                              const int k, 
                              const int nnz, 
                              const double* alpha, 
                              const double* csrVal,
                              const int* csrRowPtr, 
                              const int* csrColInd, 
                              const double* x,
                              const int ldx, 
                              const double* beta, 
                              double* y, 
                              const int ldy);

template void Cusparse::csrmm(const bool transposed,
                              const bool sym,
                              const int m, 
                              const int n, 
                              const int k, 
                              const int nnz, 
                              const float* alpha, 
                              const float* csrVal,
                              const int* csrRowPtr, 
                              const int* csrColInd, 
                              const float* x,
                              const int ldx, 
                              const float* beta, 
                              float* y, 
                              const int ldy);

 //template <typename IndexType_, typename ValueType_>
 void Cusparse::csr2coo( const int n, 
                                              const int nnz, 
                                              const int *csrRowPtr,
                                              int *cooRowInd)
 {
   hipsparseHandle_t handle = Cusparse::get_handle();
   hipsparseIndexBase_t idxBase = HIPSPARSE_INDEX_BASE_ZERO ;
   CHECK_CUSPARSE(hipsparseXcsr2coo(handle, csrRowPtr, nnz, n, cooRowInd, idxBase));

 }

} // end namespace nvgraph

