#include <mkl.h>
#include "bksp.h"
#include <stdio.h>

int dbicgstab(dspmat *mat, double *b, double tol,
                                       int max_iter, double *x) {
    int i;
    double error = 0;
    double alpha, beta, omega, fnb;
    double *r, *r_old, *rs, *s, *p, *ap, *as;
    MKL_INT n = *mat->row_size;
    r = (double *) malloc(n * sizeof(double));
    r_old = (double *) malloc(n * sizeof(double));
    rs = (double *) malloc(n * sizeof(double));
    s = (double *) malloc(n * sizeof(double));
    p = (double *) malloc(n * sizeof(double));
    ap = (double *) malloc(n * sizeof(double));
    as = (double *) malloc(n * sizeof(double));

    double *tmp = (double *) malloc(n * sizeof(double));
    mkl_cspblas_dcsrgemv("n", mat->row_size, mat->value,
                   mat->I, mat->J, x, tmp);
    cblas_dscal(n, -1, tmp, 1);
    cblas_daxpy(n, 1, b, 1, tmp, 1);
    cblas_dcopy(n, tmp, 1, r, 1);
    cblas_dcopy(n, r, 1, p, 1);
    cblas_dcopy(n, r, 1, rs, 1);
    fnb = cblas_dnrm2(n, b, 1);

    for (i = 0; i < max_iter; i++) {
        cblas_dcopy(n, r, 1, r_old, 1);
        mkl_cspblas_dcsrgemv("n", mat->row_size, mat->value,
                     mat->I, mat->J, p, ap);
        alpha = cblas_ddot(n, rs, 1, r, 1) / cblas_ddot(n, rs, 1, ap, 1);

        cblas_dcopy(n, r, 1, s, 1);
        cblas_daxpy(n, -alpha, ap, 1, s, 1);

        mkl_cspblas_dcsrgemv("n", mat->row_size, mat->value,
                    mat->I, mat->J, s, as);
        omega = cblas_ddot(n, as, 1, s, 1) / cblas_ddot(n, as, 1, as, 1);

        cblas_daxpy(n, alpha, p, 1, x, 1);
        cblas_daxpy(n, omega, s, 1, x, 1);

        cblas_dcopy(n, s, 1, r, 1);
        cblas_daxpy(n, -omega, as, 1, r, 1);

        beta = (alpha / omega) *
                cblas_ddot(n, rs, 1, r, 1) / cblas_ddot(n, rs, 1, r_old, 1);

        cblas_daxpy(n, -omega, ap, 1, p, 1);
        cblas_dscal(n, beta, p, 1);
        cblas_daxpy(n, 1, r, 1, p, 1);

        error = cblas_dnrm2(n, r, 1) / fnb;
        if (error < tol) {
            break;
        }
    }

    free(tmp);
    free(as);
    free(ap);
    free(p);
    free(s);
    free(r_old);
    return 0;
}

