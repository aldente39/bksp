
#ifdef NOMKL

#include <lapacke.h>
#include <cblas.h>

#define MKL_INT int


static inline int mkl_cspblas_dcsrsymv
(char *uplo, int *m, double *a,
 int *ia, int *ja, double *x, double *y) {

    cblas_dscal(*m, 0.0, y, 1);
    dcsrsymm0(*uplo, *m, 1, a, ia, ja, &ja[*m], x, y);

    return 0;
}

static inline int mkl_cspblas_dcsrgemv
(char *transa, int *m, double *a,
 int *ia, int *ja, double *x, double *y) {

    dcsrgemm0(*transa, *m, 1, a, ia, ja, &ja[*m], x, y);

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

#else

#include <mkl.h>

#endif

