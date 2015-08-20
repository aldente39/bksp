
#ifdef NOMKL

#ifdef OSX

#include <lapacke.h>
#define __CLAPACK_H  // cancel to include clapack.h in accelerate.h
#include <Accelerate/accelerate.h>

#else

#include <lapacke.h>
#include <cblas.h>

#endif

#define MKL_INT int

int dcsrsymm0 (char type, int m, int n, double *A,
               int *Ai, int *Aj, int nnz, double *B, double *C);
int dcsrsymm1 (char type, double alpha, int m, int n, double *A,
               int *Ai, int *Aj, int nnz, double *B, double beta, double *C);
int dcsrgemm0 (char type, int m, int n, double *A,
               int *Ai, int *Aj, int nnz, double *B, double *C);
int dcsrgemm1 (char type, double alpha, int m, int n, double *A,
               int *Ai, int *Aj, int nnz, double *B, double beta, double *C);

int zcsrgemm0 (char type, int m, int n, double _Complex*A,
               int *Ai, int *Aj, int nnz,
               double _Complex *B, double _Complex *C);
int zcsrgemm1 (char type, double _Complex *alpha,
               int m, int n, double _Complex*A, int *Ai, int *Aj, int nnz,
               double _Complex *B, double _Complex *beta, double _Complex *C);

static inline int mkl_cspblas_dcsrsymv
(char *uplo, int *m, double *a,
 int *ia, int *ja, double *x, double *y) {

    cblas_dscal(*m, 0.0, y, 1);
    dcsrsymm0(*uplo, *m, 1, a, ia, ja, ja[*m], x, y);

    return 0;
}

static inline int mkl_cspblas_dcsrgemv
(char *transa, int *m, double *a,
 int *ia, int *ja, double *x, double *y) {

    dcsrgemm0(*transa, *m, 1, a, ia, ja, ja[*m], x, y);

    return 0;
}

static inline int mkl_cspblas_zcsrgemv
(char *transa, int *m, double _Complex *a,
 int *ia, int *ja, double _Complex *x, double _Complex *y) {

    zcsrgemm0(*transa, *m, 1, a, ia, ja, ja[*m], x, y);

    return 0;
}

static inline int mkl_dcsrmm
(char *transa, int *m, int *n, int *k, double *alpha,
 char *matdescra, double *val, int *indx, int *pntrb, int *pntre,
 double *b, int *ldb, double *beta, double *c, int *ldc) {

    if (matdescra[0] == 'G') {
        dcsrgemm1(*transa, *alpha, *m, *n, val,
                pntrb, indx, pntrb[*m], b, *beta, c);
    }
    else if (matdescra[0] == 'S') {
        cblas_dscal((*m) * (*n), *beta, c, 1);
        dcsrsymm1(*transa, *alpha, *m, *n, val,
                pntrb, indx, pntrb[*m], b, *beta, c);
    }

    return 0;
}

static inline int mkl_zcsrmm
(char *transa, int *m, int *n, int *k, double _Complex *alpha,
 char *matdescra, double _Complex *val, int *indx, int *pntrb, int *pntre,
 double _Complex *b, int *ldb, double _Complex *beta,
 double _Complex *c, int *ldc) {

    if (matdescra[0] == 'G') {
        zcsrgemm1(*transa, alpha, *m, *n, val,
                  pntrb, indx, pntrb[*m], b, beta, c);
    }
    /* else if (matdescra[0] == 'S') {
     *     cblas_zscal((*m) * (*n), beta, c, 1);
     *     zcsrsymm1(*transa, *alpha, *m, *n, val,
     *               pntrb, indx, pntrb[*m], b, *beta, c);
     * } */

    return 0;
}

#ifdef OPENBLAS

#define cblas_zcopy(n, x, incx, y, incy) \
    (cblas_zcopy(n, (double *)x, incx, (double *)y, incy))
#define cblas_zdotc_sub(n, x, incx, y, incy, ret) \
    (cblas_zdotc_sub(n, (double *)x, incx, (double *)y, incy, ret))
#define cblas_zaxpy(n, a, x, incx, y, incy) \
    (cblas_zaxpy(n, (double *)a, (double *)x, incx, (double *)y, incy))
#define cblas_zgemm(Layout, transa, transb, m, n, k, \
                    alpha, a, lda, b, ldb, beta, c, ldc) \
    (cblas_zgemm(Layout, transa, transb, m, n, k, (double *)alpha, \
                 (double *)a, lda, (double *)b, ldb, \
                 (double *)beta, (double *)c, ldc))
#define cblas_dznrm2(n, x, incx) \
    (cblas_dznrm2(n, (const double *)x, incx))
#define cblas_zscal(n, a, x, incx) \
    (cblas_zscal(n, (double *)a, (double *)x, incx))
#define cblas_zdscal(n, a, x, incx) \
    (cblas_zdscal(n, a, (double *)x, incx))

#endif

#else

#define MKL_Complex16 double _Complex
#include <mkl.h>

#endif


////////// Utility functions for shifted Krylov subspace methods.

int dhorner (double *gamma, int l, double sigma, double *out);
int zhorner (double _Complex *gamma, int l,
             double _Complex sigma, double _Complex *out);
int comb (int n, int r);
