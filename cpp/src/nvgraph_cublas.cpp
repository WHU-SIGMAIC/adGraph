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
 
#include <nvgraph_cublas.hxx>

namespace nvgraph
{

hipblasHandle_t Cublas::m_handle = 0;

namespace
{
    hipblasStatus_t cublas_axpy(hipblasHandle_t handle, int n,
                               const float* alpha,
                               const float* x, int incx,
                               float* y, int incy)
    {
        return hipblasSaxpy(handle, n, alpha, x, incx, y, incy);
    }

    hipblasStatus_t cublas_axpy(hipblasHandle_t handle, int n,
                               const double* alpha,
                               const double* x, int incx,
                               double* y, int incy)
    {
        return hipblasDaxpy(handle, n, alpha, x, incx, y, incy);
    }

    hipblasStatus_t cublas_copy(hipblasHandle_t handle, int n,
                               const float* x, int incx,
                               float* y, int incy)
    {
        return hipblasScopy(handle, n, x, incx, y, incy);
    }

    hipblasStatus_t cublas_copy(hipblasHandle_t handle, int n,
                               const double* x, int incx,
                               double* y, int incy)
    {
        return hipblasDcopy(handle, n, x, incx, y, incy);
    }

    hipblasStatus_t cublas_dot(hipblasHandle_t handle, int n,
                              const float* x, int incx, const float* y, int incy,
                              float* result)
    {
        return hipblasSdot(handle, n, x, incx, y, incy, result);
    }

    hipblasStatus_t cublas_dot(hipblasHandle_t handle, int n,
                              const double* x, int incx, const double* y, int incy,
                              double* result)
    {
        return hipblasDdot(handle, n, x, incx, y, incy, result);
    }
    
    
     hipblasStatus_t cublas_trsv_v2(hipblasHandle_t handle, 
                                                      hipblasFillMode_t uplo, 
                                                      hipblasOperation_t trans, 
                                                      hipblasDiagType_t diag, 
                                                      int n, 
                                                      const float *A, 
                                                      int lda, 
                                                      float *x, 
                                                      int incx)
    {
	return hipblasStrsv (handle, uplo, trans, diag, n, A, lda, x, incx);
    }
    hipblasStatus_t cublas_trsv_v2(hipblasHandle_t handle, 
                                                      hipblasFillMode_t uplo, 
                                                      hipblasOperation_t trans, 
                                                      hipblasDiagType_t diag, 
                                                      int n, 
                                                      const double *A, 
                                                      int lda, 
                                                      double *x, 
                                                      int incx)
    {
	return hipblasDtrsv (handle, uplo, trans, diag, n, A, lda, x, incx);
    }
    
    hipblasStatus_t cublas_gemm(hipblasHandle_t handle,
                               hipblasOperation_t transa, hipblasOperation_t transb,
                               int m, int n, int k,
                               const float           *alpha,
                               const float           *A, int lda,
                               const float           *B, int ldb,
                               const float           *beta,
                               float           *C, int ldc)
    {
        return hipblasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    hipblasStatus_t cublas_gemm(hipblasHandle_t handle,
                               hipblasOperation_t transa, hipblasOperation_t transb,
                               int m, int n, int k,
                               const double          *alpha,
                               const double          *A, int lda,
                               const double          *B, int ldb,
                               const double          *beta,
                               double          *C, int ldc)
    {
        return hipblasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    hipblasStatus_t cublas_gemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n,
                               const float *alpha, const float *A, int lda,
                               const float *x, int incx,
                               const float *beta, float* y, int incy)
    {
        return hipblasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    }

    hipblasStatus_t cublas_gemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n,
                               const double *alpha, const double *A, int lda,
                               const double *x, int incx,
                               const double *beta, double* y, int incy)
    {
        return hipblasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    }

    hipblasStatus_t cublas_ger(hipblasHandle_t handle, int m, int n,
                              const float* alpha,
                              const float* x, int incx,
                              const float* y, int incy,
                              float* A, int lda)
    {
        return hipblasSger(handle, m, n, alpha, x, incx, y, incy, A, lda);
    }

    hipblasStatus_t cublas_ger(hipblasHandle_t handle, int m, int n,
                              const double* alpha,
                              const double* x, int incx,
                              const double* y, int incy,
                              double *A, int lda)
    {
        return hipblasDger(handle, m, n, alpha, x, incx, y, incy, A, lda);
    }

    hipblasStatus_t cublas_nrm2(hipblasHandle_t handle, int n,
                               const float *x, int incx, float *result)
    {
        return hipblasSnrm2(handle, n, x, incx, result);
    }

    hipblasStatus_t cublas_nrm2(hipblasHandle_t handle, int n,
                               const double *x, int incx, double *result)
    {
        return hipblasDnrm2(handle, n, x, incx, result);
    }

    hipblasStatus_t cublas_scal(hipblasHandle_t handle, int n,
                               const float* alpha,
                               float* x, int incx)
    {
        return hipblasSscal(handle, n, alpha, x, incx);
    }

    hipblasStatus_t cublas_scal(hipblasHandle_t handle, int n,
                               const double* alpha,
                               double* x, int incx)
    {
        return hipblasDscal(handle, n, alpha, x, incx);
    }

    hipblasStatus_t cublas_geam(hipblasHandle_t handle,
			       hipblasOperation_t transa,
			       hipblasOperation_t transb,
			       int m, int n,
			       const float * alpha,
			       const float * A, int lda,
			       const float * beta,
			       const float * B, int ldb,
			       float * C, int ldc) 
    {
        return hipblasSgeam(handle, transa, transb, m, n,
			   alpha, A, lda, beta, B, ldb, C, ldc);
    }

    hipblasStatus_t cublas_geam(hipblasHandle_t handle,
			       hipblasOperation_t transa,
			       hipblasOperation_t transb,
			       int m, int n,
			       const double * alpha,
			       const double * A, int lda,
			       const double * beta,
			       const double * B, int ldb,
			       double * C, int ldc) 
    {
        return hipblasDgeam(handle, transa, transb, m, n,
			   alpha, A, lda, beta, B, ldb, C, ldc);
    }
			     

} // anonymous namespace.

void Cublas::set_pointer_mode_device()
{
    hipblasHandle_t handle = Cublas::get_handle();
    hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);
}

void Cublas::set_pointer_mode_host()
{
    hipblasHandle_t handle = Cublas::get_handle();
    hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);
}

template <typename T>
void Cublas::axpy(int n, T alpha,
                  const T* x, int incx,
                  T* y, int incy)
{
    hipblasHandle_t handle = Cublas::get_handle();
    CHECK_CUBLAS(cublas_axpy(handle, n, &alpha, x, incx, y, incy));
}

template <typename T>
void Cublas::copy(int n, const T* x, int incx,
                  T* y, int incy)
{
    hipblasHandle_t handle = Cublas::get_handle();
    CHECK_CUBLAS(cublas_copy(handle, n, x, incx, y, incy));
}

template <typename T>
void Cublas::dot(int n, const T* x, int incx,
                 const T* y, int incy,
                 T* result)
{
    hipblasHandle_t handle = Cublas::get_handle();
    CHECK_CUBLAS(cublas_dot(handle, n, x, incx, y, incy, result));
}

template <typename T>
T Cublas::nrm2(int n, const T* x, int incx)
{
    Cublas::get_handle();
    T result;
    Cublas::nrm2(n, x, incx, &result);
    return result;
}

template <typename T>
void Cublas::nrm2(int n, const T* x, int incx, T* result)
{
    hipblasHandle_t handle = Cublas::get_handle();
    CHECK_CUBLAS(cublas_nrm2(handle, n, x, incx, result));
}

template <typename T>
void Cublas::scal(int n, T alpha, T* x, int incx)
{
    Cublas::scal(n, &alpha, x, incx);
}

template <typename T>
void Cublas::scal(int n, T* alpha, T* x, int incx)
{
    hipblasHandle_t handle = Cublas::get_handle();
    CHECK_CUBLAS(cublas_scal(handle, n, alpha, x, incx));
}

template <typename T>
void Cublas::gemv(bool transposed, int m, int n,
                  const T* alpha, const T* A, int lda,
                  const T* x, int incx,
                  const T* beta, T* y, int incy)
{
    hipblasHandle_t handle = Cublas::get_handle();
    hipblasOperation_t trans = transposed ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    CHECK_CUBLAS(cublas_gemv(handle, trans, m, n, alpha, A, lda,
                                 x, incx, beta, y, incy));
}

template <typename T>
void Cublas::gemv_ext(bool transposed, const int m, const int n,
                  const T* alpha, const T* A, const int lda,
                  const T* x, const int incx,
                  const T* beta, T* y, const int incy, const int offsetx, const int offsety, const int offseta)
{
    hipblasHandle_t handle = Cublas::get_handle();
    hipblasOperation_t trans = transposed ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    CHECK_CUBLAS(cublas_gemv(handle, trans, m, n, alpha, A+offseta, lda,
                                 x+offsetx, incx, beta, y+offsety, incy));
}

template <typename T>
void Cublas::trsv_v2( hipblasFillMode_t uplo, hipblasOperation_t trans, hipblasDiagType_t diag, int n, 
			      const T *A, int lda, T *x, int incx, int offseta)
{
    hipblasHandle_t handle = Cublas::get_handle();

    CHECK_CUBLAS( cublas_trsv_v2(handle, uplo, trans, diag, n, A+offseta, lda, x, incx));
}
      
      
template <typename T>
void Cublas::ger(int m, int n, const T* alpha,
                 const T* x, int incx,
                 const T* y, int incy,
                 T* A, int lda)
{
    hipblasHandle_t handle = Cublas::get_handle();
    CHECK_CUBLAS(cublas_ger(handle, m, n, alpha, x, incx, y, incy, A, lda));
}


template <typename T>
void Cublas::gemm(bool transa,
		  bool transb,
		  int m, int n, int k,
		  const T * alpha,
		  const T * A, int lda,
		  const T * B, int ldb,
		  const T * beta,
		  T * C, int ldc)
{
    hipblasHandle_t handle = Cublas::get_handle();
    hipblasOperation_t cublasTransA = transa ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasOperation_t cublasTransB = transb ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    CHECK_CUBLAS(cublas_gemm(handle, cublasTransA, cublasTransB, m, n, k,
			     alpha, A, lda, B, ldb, beta, C, ldc));
}


template <typename T>
void Cublas::geam(bool transa, bool transb, int m, int n,
		  const T * alpha, const T * A, int lda,
		  const T * beta,  const T * B, int ldb,
		  T * C, int ldc)
{
    hipblasHandle_t handle = Cublas::get_handle();
    hipblasOperation_t cublasTransA = transa ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasOperation_t cublasTransB = transb ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    CHECK_CUBLAS(cublas_geam(handle, cublasTransA, cublasTransB, m, n,
			     alpha, A, lda, beta, B, ldb, C, ldc));
}

template void Cublas::axpy(int n, float alpha,
                           const float* x, int incx,
                           float* y, int incy);
template void Cublas::axpy(int n, double alpha,
                           const double* x, int incx,
                           double* y, int incy);

template void Cublas::copy(int n, const float* x, int incx, float* y, int incy);
template void Cublas::copy(int n, const double* x, int incx, double* y, int incy);

template void Cublas::dot(int n, const float* x, int incx,
                          const float* y, int incy,
                          float* result);
template void Cublas::dot(int n, const double* x, int incx,
                          const double* y, int incy,
                          double* result);

template void Cublas::gemv(bool transposed, int m, int n,
                           const float* alpha, const float* A, int lda,
                           const float* x, int incx,
                           const float* beta, float* y, int incy);
template void Cublas::gemv(bool transposed, int m, int n,
                           const double* alpha, const double* A, int lda,
                           const double* x, int incx,
                           const double* beta, double* y, int incy);

template void Cublas::ger(int m, int n, const float* alpha,
                          const float* x, int incx,
                          const float* y, int incy,
                          float* A, int lda);
template void Cublas::ger(int m, int n, const double* alpha,
                          const double* x, int incx,
                          const double* y, int incy,
                          double* A, int lda);


template void Cublas::gemv_ext(bool transposed, const int m, const int n,
                           const float* alpha, const float* A, const int lda,
                           const float* x, const int incx,
                           const float* beta, float* y, const int incy, const int offsetx, const int offsety, const int offseta);
template void Cublas::gemv_ext(bool transposed, const int m, const int n,
                           const double* alpha, const double* A, const int lda,
                           const double* x, const int incx,
                           const double* beta, double* y, const int incy, const int offsetx, const int offsety, const int offseta);


template void Cublas::trsv_v2( hipblasFillMode_t uplo, hipblasOperation_t trans, hipblasDiagType_t diag, int n, 
			      const float *A, int lda, float *x, int incx, int offseta);
template void Cublas::trsv_v2( hipblasFillMode_t uplo, hipblasOperation_t trans, hipblasDiagType_t diag, int n, 
			      const double *A, int lda, double *x, int incx, int offseta);

template double Cublas::nrm2(int n, const double* x, int incx);
template float Cublas::nrm2(int n, const float* x, int incx);

template void Cublas::scal(int n, float alpha, float* x, int incx);
template void Cublas::scal(int n, double alpha, double* x, int incx);

template void Cublas::gemm(bool transa, bool transb,
			   int m, int n, int k,
			   const float * alpha,
			   const float * A, int lda,
			   const float * B, int ldb,
			   const float * beta,
			   float * C, int ldc);
template void Cublas::gemm(bool transa, bool transb,
			   int m, int n, int k,
			   const double * alpha,
			   const double * A, int lda,
			   const double * B, int ldb,
			   const double * beta,
			   double * C, int ldc);

template void Cublas::geam(bool transa, bool transb, int m, int n,
			   const float * alpha, const float * A, int lda,
			   const float * beta,  const float * B, int ldb,
			   float * C, int ldc);
template void Cublas::geam(bool transa, bool transb, int m, int n,
			   const double * alpha, const double * A, int lda,
			   const double * beta,  const double * B, int ldb,
			   double * C, int ldc);


} // end namespace nvgraph

