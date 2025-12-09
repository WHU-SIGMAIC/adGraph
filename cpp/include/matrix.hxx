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
#pragma once

#include <hip/hip_runtime.h>
#include <hipblas.h>
#include <hiprand.h>
#include <hipsolver.h>
#include <hipsparse.h>

#include "nvgraph_vector.hxx"
#include "valued_csr_graph.hxx"

namespace nvgraph
{

  /// Abstract matrix class
  /** Derived classes must implement matrix-vector products.
   */
  template <typename IndexType_, typename ValueType_>
  class Matrix
  {
  public:
    /// Number of rows
    const IndexType_ m;
    /// Number of columns
    const IndexType_ n;
    /// CUDA stream
    hipStream_t s;

    /// Constructor
    /** @param _m Number of rows.
     *  @param _n Number of columns.
     */
    Matrix(IndexType_ _m, IndexType_ _n) : m(_m), n(_n), s(0) {}

    /// Destructor
    virtual ~Matrix() {}

    /// Get and Set CUDA stream
    virtual void setCUDAStream(hipStream_t _s) = 0;
    virtual void getCUDAStream(hipStream_t *_s) = 0;

    /// Matrix-vector product
    /** y is overwritten with alpha*A*x+beta*y.
     *
     *  @param alpha Scalar.
     *  @param x (Input, device memory, n entries) Vector.
     *  @param beta Scalar.
     *  @param y (Input/output, device memory, m entries) Output
     *    vector.
     */
    virtual void mv(ValueType_ alpha,
                    const ValueType_ *__restrict__ x,
                    ValueType_ beta,
                    ValueType_ *__restrict__ y) const = 0;

    virtual void mm(IndexType_ k, ValueType_ alpha, const ValueType_ *__restrict__ x, ValueType_ beta, ValueType_ *__restrict__ y) const = 0;
    /// Color and Reorder
    virtual void color(IndexType_ *c, IndexType_ *p) const = 0;
    virtual void reorder(IndexType_ *p) const = 0;

    /// Incomplete Cholesky (setup, factor and solve)
    virtual void prec_setup(Matrix<IndexType_, ValueType_> *_M) = 0;
    virtual void prec_solve(IndexType_ k, ValueType_ alpha, ValueType_ *__restrict__ fx, ValueType_ *__restrict__ t) const = 0;

    // Get the sum of all edges
    virtual ValueType_ getEdgeSum() const = 0;
  };

  /// Dense matrix class
  template <typename IndexType_, typename ValueType_>
  class DenseMatrix : public Matrix<IndexType_, ValueType_>
  {

  private:
    /// Whether to transpose matrix
    const bool trans;
    /// Matrix entries, stored column-major in device memory
    const ValueType_ *A;
    /// Leading dimension of matrix entry array
    const IndexType_ lda;

  public:
    /// Constructor
    DenseMatrix(bool _trans,
                IndexType_ _m, IndexType_ _n,
                const ValueType_ *_A, IndexType_ _lda);

    /// Destructor
    virtual ~DenseMatrix();

    /// Get and Set CUDA stream
    virtual void setCUDAStream(hipStream_t _s);
    virtual void getCUDAStream(hipStream_t *_s);

    /// Matrix-vector product
    virtual void mv(ValueType_ alpha, const ValueType_ *__restrict__ x,
                    ValueType_ beta, ValueType_ *__restrict__ y) const;
    /// Matrix-set of k vectors product
    virtual void mm(IndexType_ k, ValueType_ alpha, const ValueType_ *__restrict__ x, ValueType_ beta, ValueType_ *__restrict__ y) const;

    /// Color and Reorder
    virtual void color(IndexType_ *c, IndexType_ *p) const;
    virtual void reorder(IndexType_ *p) const;

    /// Incomplete Cholesky (setup, factor and solve)
    virtual void prec_setup(Matrix<IndexType_, ValueType_> *_M);
    virtual void prec_solve(IndexType_ k, ValueType_ alpha, ValueType_ *__restrict__ fx, ValueType_ *__restrict__ t) const;

    // Get the sum of all edges
    virtual ValueType_ getEdgeSum() const;
  };

  /// Sparse matrix class in CSR format
  template <typename IndexType_, typename ValueType_>
  class CsrMatrix : public Matrix<IndexType_, ValueType_>
  {

  private:
    /// Whether to transpose matrix
    const bool trans;
    /// Whether matrix is stored in symmetric format
    const bool sym;
    /// Number of non-zero entries
    const IndexType_ nnz;
    /// Matrix properties
    const hipsparseMatDescr_t descrA;
    /// Matrix entry values (device memory)
    /*const*/ ValueType_ *csrValA;
    /// Pointer to first entry in each row (device memory)
    const IndexType_ *csrRowPtrA;
    /// Column index of each matrix entry (device memory)
    const IndexType_ *csrColIndA;
    /// Analysis info (pointer to opaque CUSPARSE struct)
    // cusparseSolveAnalysisInfo_t info_l;
    // cusparseSolveAnalysisInfo_t info_u;
    /// factored flag (originally set to false, then reset to true after factorization),
    /// notice we only want to factor once
    bool factored;

  public:
    /// Constructor
    CsrMatrix(bool _trans, bool _sym,
              IndexType_ _m, IndexType_ _n, IndexType_ _nnz,
              const hipsparseMatDescr_t _descrA,
              /*const*/ ValueType_ *_csrValA,
              const IndexType_ *_csrRowPtrA,
              const IndexType_ *_csrColIndA);

    /// Constructor
    CsrMatrix(ValuedCsrGraph<IndexType_, ValueType_> &G, const hipsparseMatDescr_t _descrA = 0);

    /// Destructor
    virtual ~CsrMatrix();

    /// Get and Set CUDA stream
    virtual void setCUDAStream(hipStream_t _s);
    virtual void getCUDAStream(hipStream_t *_s);

    /// Matrix-vector product
    virtual void mv(ValueType_ alpha, const ValueType_ *__restrict__ x,
                    ValueType_ beta, ValueType_ *__restrict__ y) const;
    /// Matrix-set of k vectors product
    virtual void mm(IndexType_ k, ValueType_ alpha, const ValueType_ *__restrict__ x, ValueType_ beta, ValueType_ *__restrict__ y) const;

    /// Color and Reorder
    virtual void color(IndexType_ *c, IndexType_ *p) const;
    virtual void reorder(IndexType_ *p) const;

    /// Incomplete Cholesky (setup, factor and solve)
    virtual void prec_setup(Matrix<IndexType_, ValueType_> *_M);
    virtual void prec_solve(IndexType_ k, ValueType_ alpha, ValueType_ *__restrict__ fx, ValueType_ *__restrict__ t) const;

    // Get the sum of all edges
    virtual ValueType_ getEdgeSum() const;
  };

  /// Graph Laplacian matrix
  template <typename IndexType_, typename ValueType_>
  class LaplacianMatrix
      : public Matrix<IndexType_, ValueType_>
  {

  private:
    /// Adjacency matrix
    /*const*/ Matrix<IndexType_, ValueType_> *A;
    /// Degree of each vertex
    Vector<ValueType_> D;
    /// Preconditioning matrix
    Matrix<IndexType_, ValueType_> *M;

  public:
    /// Constructor
    LaplacianMatrix(/*const*/ Matrix<IndexType_, ValueType_> &_A);

    /// Destructor
    virtual ~LaplacianMatrix();

    /// Get and Set CUDA stream
    virtual void setCUDAStream(hipStream_t _s);
    virtual void getCUDAStream(hipStream_t *_s);

    /// Matrix-vector product
    virtual void mv(ValueType_ alpha, const ValueType_ *__restrict__ x,
                    ValueType_ beta, ValueType_ *__restrict__ y) const;
    /// Matrix-set of k vectors product
    virtual void mm(IndexType_ k, ValueType_ alpha, const ValueType_ *__restrict__ x, ValueType_ beta, ValueType_ *__restrict__ y) const;

    /// Scale a set of k vectors by a diagonal
    virtual void dm(IndexType_ k, ValueType_ alpha, const ValueType_ *__restrict__ x, ValueType_ beta, ValueType_ *__restrict__ y) const;

    /// Color and Reorder
    virtual void color(IndexType_ *c, IndexType_ *p) const;
    virtual void reorder(IndexType_ *p) const;

    /// Solve preconditioned system M x = f for a set of k vectors
    virtual void prec_setup(Matrix<IndexType_, ValueType_> *_M);
    virtual void prec_solve(IndexType_ k, ValueType_ alpha, ValueType_ *__restrict__ fx, ValueType_ *__restrict__ t) const;

    // Get the sum of all edges
    virtual ValueType_ getEdgeSum() const;
  };

  ///  Modularity matrix
  template <typename IndexType_, typename ValueType_>
  class ModularityMatrix
      : public Matrix<IndexType_, ValueType_>
  {

  private:
    /// Adjacency matrix
    /*const*/ Matrix<IndexType_, ValueType_> *A;
    /// Degree of each vertex
    Vector<ValueType_> D;
    IndexType_ nnz;
    ValueType_ edge_sum;

    /// Preconditioning matrix
    Matrix<IndexType_, ValueType_> *M;

  public:
    /// Constructor
    ModularityMatrix(/*const*/ Matrix<IndexType_, ValueType_> &_A, IndexType_ _nnz);

    /// Destructor
    virtual ~ModularityMatrix();

    /// Get and Set CUDA stream
    virtual void setCUDAStream(hipStream_t _s);
    virtual void getCUDAStream(hipStream_t *_s);

    /// Matrix-vector product
    virtual void mv(ValueType_ alpha, const ValueType_ *__restrict__ x,
                    ValueType_ beta, ValueType_ *__restrict__ y) const;
    /// Matrix-set of k vectors product
    virtual void mm(IndexType_ k, ValueType_ alpha, const ValueType_ *__restrict__ x, ValueType_ beta, ValueType_ *__restrict__ y) const;

    /// Scale a set of k vectors by a diagonal
    virtual void dm(IndexType_ k, ValueType_ alpha, const ValueType_ *__restrict__ x, ValueType_ beta, ValueType_ *__restrict__ y) const;

    /// Color and Reorder
    virtual void color(IndexType_ *c, IndexType_ *p) const;
    virtual void reorder(IndexType_ *p) const;

    /// Solve preconditioned system M x = f for a set of k vectors
    virtual void prec_setup(Matrix<IndexType_, ValueType_> *_M);
    virtual void prec_solve(IndexType_ k, ValueType_ alpha, ValueType_ *__restrict__ fx, ValueType_ *__restrict__ t) const;

    // Get the sum of all edges
    virtual ValueType_ getEdgeSum() const;
  };

  // cublasIxamax
  inline hipblasStatus_t cublasIxamax(hipblasHandle_t handle, int n,
                                      const float *x, int incx, int *result)
  {
    return hipblasIsamax(handle, n, x, incx, result);
  }
  inline hipblasStatus_t cublasIxamax(hipblasHandle_t handle, int n,
                                      const double *x, int incx, int *result)
  {
    return hipblasIdamax(handle, n, x, incx, result);
  }

  // cublasIxamin
  inline hipblasStatus_t cublasIxamin(hipblasHandle_t handle, int n,
                                      const float *x, int incx, int *result)
  {
    return hipblasIsamin(handle, n, x, incx, result);
  }
  inline hipblasStatus_t cublasIxamin(hipblasHandle_t handle, int n,
                                      const double *x, int incx, int *result)
  {
    return hipblasIdamin(handle, n, x, incx, result);
  }

  // cublasXasum
  inline hipblasStatus_t cublasXasum(hipblasHandle_t handle, int n,
                                     const float *x, int incx,
                                     float *result)
  {
    return hipblasSasum(handle, n, x, incx, result);
  }
  inline hipblasStatus_t cublasXasum(hipblasHandle_t handle, int n,
                                     const double *x, int incx,
                                     double *result)
  {
    return hipblasDasum(handle, n, x, incx, result);
  }

  // cublasXaxpy
  inline hipblasStatus_t cublasXaxpy(hipblasHandle_t handle, int n,
                                     const float *alpha,
                                     const float *x, int incx,
                                     float *y, int incy)
  {
    return hipblasSaxpy(handle, n, alpha, x, incx, y, incy);
  }
  inline hipblasStatus_t cublasXaxpy(hipblasHandle_t handle, int n,
                                     const double *alpha,
                                     const double *x, int incx,
                                     double *y, int incy)
  {
    return hipblasDaxpy(handle, n, alpha, x, incx, y, incy);
  }

  // cublasXcopy
  inline hipblasStatus_t cublasXcopy(hipblasHandle_t handle, int n,
                                     const float *x, int incx,
                                     float *y, int incy)
  {
    return hipblasScopy(handle, n, x, incx, y, incy);
  }
  inline hipblasStatus_t cublasXcopy(hipblasHandle_t handle, int n,
                                     const double *x, int incx,
                                     double *y, int incy)
  {
    return hipblasDcopy(handle, n, x, incx, y, incy);
  }

  // cublasXdot
  inline hipblasStatus_t cublasXdot(hipblasHandle_t handle, int n,
                                    const float *x, int incx,
                                    const float *y, int incy,
                                    float *result)
  {
    return hipblasSdot(handle, n, x, incx, y, incy, result);
  }
  inline hipblasStatus_t cublasXdot(hipblasHandle_t handle, int n,
                                    const double *x, int incx,
                                    const double *y, int incy,
                                    double *result)
  {
    return hipblasDdot(handle, n, x, incx, y, incy, result);
  }

  // cublasXnrm2
  inline hipblasStatus_t cublasXnrm2(hipblasHandle_t handle, int n,
                                     const float *x, int incx,
                                     float *result)
  {
    return hipblasSnrm2(handle, n, x, incx, result);
  }
  inline hipblasStatus_t cublasXnrm2(hipblasHandle_t handle, int n,
                                     const double *x, int incx,
                                     double *result)
  {
    return hipblasDnrm2(handle, n, x, incx, result);
  }

  // cublasXscal
  inline hipblasStatus_t cublasXscal(hipblasHandle_t handle, int n,
                                     const float *alpha,
                                     float *x, int incx)
  {
    return hipblasSscal(handle, n, alpha, x, incx);
  }
  inline hipblasStatus_t cublasXscal(hipblasHandle_t handle, int n,
                                     const double *alpha,
                                     double *x, int incx)
  {
    return hipblasDscal(handle, n, alpha, x, incx);
  }

  // cublasXgemv
  inline hipblasStatus_t cublasXgemv(hipblasHandle_t handle,
                                     hipblasOperation_t trans,
                                     int m, int n,
                                     const float *alpha,
                                     const float *A, int lda,
                                     const float *x, int incx,
                                     const float *beta,
                                     float *y, int incy)
  {
    return hipblasSgemv(handle, trans, m, n, alpha, A, lda, x, incx,
                        beta, y, incy);
  }
  inline hipblasStatus_t cublasXgemv(hipblasHandle_t handle,
                                     hipblasOperation_t trans,
                                     int m, int n,
                                     const double *alpha,
                                     const double *A, int lda,
                                     const double *x, int incx,
                                     const double *beta,
                                     double *y, int incy)
  {
    return hipblasDgemv(handle, trans, m, n, alpha, A, lda, x, incx,
                        beta, y, incy);
  }

  // cublasXger
  inline hipblasStatus_t cublasXger(hipblasHandle_t handle, int m, int n,
                                    const float *alpha,
                                    const float *x, int incx,
                                    const float *y, int incy,
                                    float *A, int lda)
  {
    return hipblasSger(handle, m, n, alpha, x, incx, y, incy, A, lda);
  }
  inline hipblasStatus_t cublasXger(hipblasHandle_t handle, int m, int n,
                                    const double *alpha,
                                    const double *x, int incx,
                                    const double *y, int incy,
                                    double *A, int lda)
  {
    return hipblasDger(handle, m, n, alpha, x, incx, y, incy, A, lda);
  }

  // cublasXgemm
  inline hipblasStatus_t cublasXgemm(hipblasHandle_t handle,
                                     hipblasOperation_t transa,
                                     hipblasOperation_t transb,
                                     int m, int n, int k,
                                     const float *alpha,
                                     const float *A, int lda,
                                     const float *B, int ldb,
                                     const float *beta,
                                     float *C, int ldc)
  {
    return hipblasSgemm(handle, transa, transb, m, n, k,
                        alpha, A, lda, B, ldb, beta, C, ldc);
  }
  inline hipblasStatus_t cublasXgemm(hipblasHandle_t handle,
                                     hipblasOperation_t transa,
                                     hipblasOperation_t transb,
                                     int m, int n, int k,
                                     const double *alpha,
                                     const double *A, int lda,
                                     const double *B, int ldb,
                                     const double *beta,
                                     double *C, int ldc)
  {
    return hipblasDgemm(handle, transa, transb, m, n, k,
                        alpha, A, lda, B, ldb, beta, C, ldc);
  }

  // cublasXgeam
  inline hipblasStatus_t cublasXgeam(hipblasHandle_t handle,
                                     hipblasOperation_t transa,
                                     hipblasOperation_t transb,
                                     int m, int n,
                                     const float *alpha,
                                     const float *A, int lda,
                                     const float *beta,
                                     const float *B, int ldb,
                                     float *C, int ldc)
  {
    return hipblasSgeam(handle, transa, transb, m, n,
                        alpha, A, lda, beta, B, ldb, C, ldc);
  }
  inline hipblasStatus_t cublasXgeam(hipblasHandle_t handle,
                                     hipblasOperation_t transa,
                                     hipblasOperation_t transb,
                                     int m, int n,
                                     const double *alpha,
                                     const double *A, int lda,
                                     const double *beta,
                                     const double *B, int ldb,
                                     double *C, int ldc)
  {
    return hipblasDgeam(handle, transa, transb, m, n,
                        alpha, A, lda, beta, B, ldb, C, ldc);
  }

  // cublasXtrsm
  inline hipblasStatus_t cublasXtrsm(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t trans, hipblasDiagType_t diag, int m, int n, const float *alpha, float *A, int lda, float *B, int ldb)
  {
    return hipblasStrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
  }
  inline hipblasStatus_t cublasXtrsm(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t trans, hipblasDiagType_t diag, int m, int n, const double *alpha, double *A, int lda, double *B, int ldb)
  {
    return hipblasDtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
  }

  // curandGeneratorNormalX
  inline hiprandStatus_t
  curandGenerateNormalX(hiprandGenerator_t generator,
                        float *outputPtr, size_t n,
                        float mean, float stddev)
  {
    return hiprandGenerateNormal(generator, outputPtr, n, mean, stddev);
  }
  inline hiprandStatus_t
  curandGenerateNormalX(hiprandGenerator_t generator,
                        double *outputPtr, size_t n,
                        double mean, double stddev)
  {
    return hiprandGenerateNormalDouble(generator, outputPtr,
                                       n, mean, stddev);
  }

  // cusolverXpotrf_bufferSize
  inline hipsolverStatus_t cusolverXpotrf_bufferSize(hipsolverDnHandle_t handle, int n, float *A, int lda, int *Lwork)
  {
    return hipsolverDnSpotrf_bufferSize(handle, HIPSOLVER_FILL_MODE_LOWER, n, A, lda, Lwork);
  }
  inline hipsolverStatus_t cusolverXpotrf_bufferSize(hipsolverDnHandle_t handle, int n, double *A, int lda, int *Lwork)
  {
    return hipsolverDnDpotrf_bufferSize(handle, HIPSOLVER_FILL_MODE_LOWER, n, A, lda, Lwork);
  }

  // cusolverXpotrf
  inline hipsolverStatus_t cusolverXpotrf(hipsolverDnHandle_t handle, int n, float *A, int lda, float *Workspace, int Lwork, int *devInfo)
  {
    return hipsolverDnSpotrf(handle, HIPSOLVER_FILL_MODE_LOWER, n, A, lda, Workspace, Lwork, devInfo);
  }
  inline hipsolverStatus_t cusolverXpotrf(hipsolverDnHandle_t handle, int n, double *A, int lda, double *Workspace, int Lwork, int *devInfo)
  {
    return hipsolverDnDpotrf(handle, HIPSOLVER_FILL_MODE_LOWER, n, A, lda, Workspace, Lwork, devInfo);
  }

  // cusolverXgesvd_bufferSize
  inline hipsolverStatus_t cusolverXgesvd_bufferSize(hipsolverDnHandle_t handle, int m, int n, float *A, int lda, float *U, int ldu, float *VT, int ldvt, int *Lwork)
  {
    // ideally
    // char jobu = 'O';
    // char jobvt= 'N';
    // only supported
    // char jobu = 'A';
    // char jobvt= 'A';
    return hipsolverDnSgesvd_bufferSize(handle, m, n, Lwork);
  }

  inline hipsolverStatus_t cusolverXgesvd_bufferSize(hipsolverDnHandle_t handle, int m, int n, double *A, int lda, double *U, int ldu, double *VT, int ldvt, int *Lwork)
  {
    // ideally
    // char jobu = 'O';
    // char jobvt= 'N';
    // only supported
    // char jobu = 'A';
    // char jobvt= 'A';
    return hipsolverDnDgesvd_bufferSize(handle, m, n, Lwork);
  }

  // cusolverXgesvd
  inline hipsolverStatus_t cusolverXgesvd(hipsolverDnHandle_t handle, int m, int n, float *A, int lda, float *S, float *U, int ldu, float *VT, int ldvt, float *Work, int Lwork, float *rwork, int *devInfo)
  {
    // ideally
    // char jobu = 'O';
    // char jobvt= 'N';
    // only supported
    char jobu = 'A';
    char jobvt = 'A';

    return hipsolverDnSgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, Work, Lwork, rwork, devInfo);
  }

  inline hipsolverStatus_t cusolverXgesvd(hipsolverDnHandle_t handle, int m, int n, double *A, int lda, double *S, double *U, int ldu, double *VT, int ldvt, double *Work, int Lwork, double *rwork, int *devInfo)
  {
    // ideally
    // char jobu = 'O';
    // char jobvt= 'N';
    // only supported
    char jobu = 'A';
    char jobvt = 'A';
    return hipsolverDnDgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, Work, Lwork, rwork, devInfo);
  }

  // cusolverXgesvd_cond
  inline hipsolverStatus_t cusolverXgesvd_cond(hipsolverDnHandle_t handle, int m, int n, float *A, int lda, float *S, float *U, int ldu, float *VT, int ldvt, float *Work, int Lwork, float *rwork, int *devInfo)
  {
    // ideally
    // char jobu = 'N';
    // char jobvt= 'N';
    // only supported
    char jobu = 'A';
    char jobvt = 'A';
    return hipsolverDnSgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, Work, Lwork, rwork, devInfo);
  }

  inline hipsolverStatus_t cusolverXgesvd_cond(hipsolverDnHandle_t handle, int m, int n, double *A, int lda, double *S, double *U, int ldu, double *VT, int ldvt, double *Work, int Lwork, double *rwork, int *devInfo)
  {
    // ideally
    // char jobu = 'N';
    // char jobvt= 'N';
    // only supported
    char jobu = 'A';
    char jobvt = 'A';
    return hipsolverDnDgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, Work, Lwork, rwork, devInfo);
  }

  // cusparseXcsrmv
  inline hipsparseStatus_t cusparseXcsrmv(hipsparseHandle_t handle,
                                          hipsparseOperation_t transA,
                                          int m, int n, int nnz,
                                          const float *alpha,
                                          const hipsparseMatDescr_t descrA,
                                          const float *csrValA,
                                          const int *csrRowPtrA,
                                          const int *csrColIndA,
                                          const float *x,
                                          const float *beta,
                                          float *y)
  {
    return hipsparseScsrmv(handle, transA, m, n, nnz,
                             alpha, descrA, csrValA, csrRowPtrA, csrColIndA,
                             x, beta, y);
  }
  inline hipsparseStatus_t cusparseXcsrmv(hipsparseHandle_t handle,
                                          hipsparseOperation_t transA,
                                          int m, int n, int nnz,
                                          const double *alpha,
                                          const hipsparseMatDescr_t descrA,
                                          const double *csrValA,
                                          const int *csrRowPtrA,
                                          const int *csrColIndA,
                                          const double *x,
                                          const double *beta,
                                          double *y)
  {
    return hipsparseDcsrmv(handle, transA, m, n, nnz,
                             alpha, descrA, csrValA, csrRowPtrA, csrColIndA,
                             x, beta, y);
  }

  // cusparseXcsrmm
  inline hipsparseStatus_t cusparseXcsrmm(hipsparseHandle_t handle,
                                          hipsparseOperation_t transA,
                                          int m, int n, int k, int nnz,
                                          const float *alpha,
                                          const hipsparseMatDescr_t descrA,
                                          const float *csrValA,
                                          const int *csrRowPtrA,
                                          const int *csrColIndA,
                                          const float *B, int ldb,
                                          const float *beta,
                                          float *C, int ldc)
  {
    return hipsparseScsrmm(handle, transA, m, n, k, nnz,
                           alpha, descrA, csrValA,
                           csrRowPtrA, csrColIndA,
                           B, ldb, beta, C, ldc);
  }
  inline hipsparseStatus_t cusparseXcsrmm(hipsparseHandle_t handle,
                                          hipsparseOperation_t transA,
                                          int m, int n, int k, int nnz,
                                          const double *alpha,
                                          const hipsparseMatDescr_t descrA,
                                          const double *csrValA,
                                          const int *csrRowPtrA,
                                          const int *csrColIndA,
                                          const double *B, int ldb,
                                          const double *beta,
                                          double *C, int ldc)
  {
    return hipsparseDcsrmm(handle, transA, m, n, k, nnz,
                           alpha, descrA, csrValA,
                           csrRowPtrA, csrColIndA,
                           B, ldb, beta, C, ldc);
  }

  // cusparseXcsrgeam
  inline hipsparseStatus_t cusparseXcsrgeam(hipsparseHandle_t handle,
                                            int m, int n,
                                            const float *alpha,
                                            const hipsparseMatDescr_t descrA,
                                            int nnzA, const float *csrValA,
                                            const int *csrRowPtrA,
                                            const int *csrColIndA,
                                            const float *beta,
                                            const hipsparseMatDescr_t descrB,
                                            int nnzB, const float *csrValB,
                                            const int *csrRowPtrB,
                                            const int *csrColIndB,
                                            const hipsparseMatDescr_t descrC,
                                            float *csrValC,
                                            int *csrRowPtrC, int *csrColIndC)
  {
    return hipsparseScsrgeam(handle, m, n,
                             alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA,
                             beta, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB,
                             descrC, csrValC, csrRowPtrC, csrColIndC);
  }
  inline hipsparseStatus_t cusparseXcsrgeam(hipsparseHandle_t handle,
                                            int m, int n,
                                            const double *alpha,
                                            const hipsparseMatDescr_t descrA,
                                            int nnzA, const double *csrValA,
                                            const int *csrRowPtrA,
                                            const int *csrColIndA,
                                            const double *beta,
                                            const hipsparseMatDescr_t descrB,
                                            int nnzB, const double *csrValB,
                                            const int *csrRowPtrB,
                                            const int *csrColIndB,
                                            const hipsparseMatDescr_t descrC,
                                            double *csrValC,
                                            int *csrRowPtrC, int *csrColIndC)
  {
    return hipsparseDcsrgeam(handle, m, n,
                             alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA,
                             beta, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB,
                             descrC, csrValC, csrRowPtrC, csrColIndC);
  }

  // ILU0, incomplete-LU with 0 threshhold (CUSPARSE)
  // inline hipsparseStatus_t cusparseXcsrilu0(hipsparseHandle_t handle,
  //                                           hipsparseOperation_t trans,
  //                                           int m,
  //                                           const hipsparseMatDescr_t descrA,
  //                                           float *csrValM,
  //                                           const int *csrRowPtrA,
  //                                           const int *csrColIndA,
  //                                           cusparseSolveAnalysisInfo_t info)
  // {
  //   return cusparseScsrilu0(handle, trans, m, descrA, csrValM, csrRowPtrA, csrColIndA, info);
  // }

  // inline hipsparseStatus_t cusparseXcsrilu0(hipsparseHandle_t handle,
  //                                           hipsparseOperation_t trans,
  //                                           int m,
  //                                           const hipsparseMatDescr_t descrA,
  //                                           double *csrValM,
  //                                           const int *csrRowPtrA,
  //                                           const int *csrColIndA,
  //                                           cusparseSolveAnalysisInfo_t info)
  // {
  //   return cusparseDcsrilu0(handle, trans, m, descrA, csrValM, csrRowPtrA, csrColIndA, info);
  // }

  // IC0, incomplete-Cholesky with 0 threshhold (CUSPARSE)
  // inline hipsparseStatus_t cusparseXcsric0(hipsparseHandle_t handle,
  //                                          hipsparseOperation_t trans,
  //                                          int m,
  //                                          const hipsparseMatDescr_t descrA,
  //                                          float *csrValM,
  //                                          const int *csrRowPtrA,
  //                                          const int *csrColIndA,
  //                                          cusparseSolveAnalysisInfo_t info)
  // {
  //   return cusparseScsric0(handle, trans, m, descrA, csrValM, csrRowPtrA, csrColIndA, info);
  // }
  // inline hipsparseStatus_t cusparseXcsric0(hipsparseHandle_t handle,
  //                                          hipsparseOperation_t trans,
  //                                          int m,
  //                                          const hipsparseMatDescr_t descrA,
  //                                          double *csrValM,
  //                                          const int *csrRowPtrA,
  //                                          const int *csrColIndA,
  //                                          cusparseSolveAnalysisInfo_t info)
  // {
  //   return cusparseDcsric0(handle, trans, m, descrA, csrValM, csrRowPtrA, csrColIndA, info);
  // }

  // sparse triangular solve (CUSPARSE)
  // analysis phase
  // inline hipsparseStatus_t cusparseXcsrsm_analysis(hipsparseHandle_t handle, hipsparseOperation_t transa, int m, int nnz, const hipsparseMatDescr_t descra,
  //                                                  const float *a, const int *ia, const int *ja, cusparseSolveAnalysisInfo_t info)
  // {
  //   return cusparseScsrsm_analysis(handle, transa, m, nnz, descra, a, ia, ja, info);
  // }
  // inline hipsparseStatus_t cusparseXcsrsm_analysis(hipsparseHandle_t handle, hipsparseOperation_t transa, int m, int nnz, const hipsparseMatDescr_t descra,
  //                                                  const double *a, const int *ia, const int *ja, cusparseSolveAnalysisInfo_t info)
  // {
  //   return cusparseDcsrsm_analysis(handle, transa, m, nnz, descra, a, ia, ja, info);
  // }
  // solve phase
  // inline hipsparseStatus_t cusparseXcsrsm_solve(hipsparseHandle_t handle, hipsparseOperation_t transa, int m, int k, float alpha, const hipsparseMatDescr_t descra,
  //                                               const float *a, const int *ia, const int *ja, cusparseSolveAnalysisInfo_t info, const float *x, int ldx, float *y, int ldy)
  // {
  //   return cusparseScsrsm_solve(handle, transa, m, k, &alpha, descra, a, ia, ja, info, x, ldx, y, ldy);
  // }
  // inline hipsparseStatus_t cusparseXcsrsm_solve(hipsparseHandle_t handle, hipsparseOperation_t transa, int m, int k, double alpha, const hipsparseMatDescr_t descra,
  //                                               const double *a, const int *ia, const int *ja, cusparseSolveAnalysisInfo_t info, const double *x, int ldx, double *y, int ldy)
  // {
  //   return cusparseDcsrsm_solve(handle, transa, m, k, &alpha, descra, a, ia, ja, info, x, ldx, y, ldy);
  // }

  inline hipsparseStatus_t cusparseXcsrcolor(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, const float *csrValA, const int *csrRowPtrA, const int *csrColIndA, const float *fractionToColor, int *ncolors, int *coloring, int *reordering, hipsparseColorInfo_t info)
  {
    return hipsparseScsrcolor(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, fractionToColor, ncolors, coloring, reordering, info);
  }
  inline hipsparseStatus_t cusparseXcsrcolor(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, const double *csrValA, const int *csrRowPtrA, const int *csrColIndA, const double *fractionToColor, int *ncolors, int *coloring, int *reordering, hipsparseColorInfo_t info)
  {
    return hipsparseDcsrcolor(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, fractionToColor, ncolors, coloring, reordering, info);
  }

}
