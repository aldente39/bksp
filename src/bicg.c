#include <stdio.h>
#include <mkl.h>
#include "bksp.h"

int dbicg (dspmat *mat, double *b,
            double tol, int max_iter, double *x) {
    int i;
    double error = 0;
    double alpha, beta, fnb;
    double *r, *r_old, *sr, *sr_old, *p, *sp, *ap, *atp;
    MKL_INT n = *mat->row_size;
    r = (double *) malloc(n * sizeof(double));
    r_old = (double *) malloc(n * sizeof(double));
    sr = (double *) malloc(n * sizeof(double));
    sr_old = (double *) malloc(n * sizeof(double));
    p = (double *) malloc(n * sizeof(double));
    sp = (double *) malloc(n * sizeof(double));
    ap = (double *) malloc(n * sizeof(double));
    atp = (double *) malloc(n * sizeof(double));

    double *tmp = (double *) malloc(n * sizeof(double));
    mkl_cspblas_dcsrgemv("n", mat->row_size, mat->value,
                   mat->I, mat->J, x, tmp);
    cblas_dscal(n, -1, tmp, 1);
    cblas_daxpy(n, 1, b, 1, tmp, 1);
    cblas_dcopy(n, tmp, 1, r, 1);
    cblas_dcopy(n, tmp, 1, sr, 1);
    cblas_dcopy(n, r, 1, p, 1);
    cblas_dcopy(n, sr, 1, sp, 1);
    fnb = cblas_dnrm2(n, b, 1);

    for (i = 0; i < max_iter; i++) {
        cblas_dcopy(n, r, 1, r_old, 1);
        cblas_dcopy(n, sr, 1, sr_old, 1);
        mkl_cspblas_dcsrgemv("n", mat->row_size, mat->value,
                     mat->I, mat->J, p, ap);
        mkl_cspblas_dcsrgemv("t", mat->row_size, mat->value,
                     mat->I, mat->J, sp, atp);
        alpha = cblas_ddot(n, r, 1, sr, 1) / cblas_ddot(n, ap, 1, sp, 1);
        cblas_daxpy(n, alpha, p, 1, x, 1);
        cblas_daxpy(n, -alpha, ap, 1, r, 1);
        cblas_daxpy(n, -alpha, atp, 1, sr, 1);
        beta = cblas_ddot(n, r, 1, sr, 1)
               / cblas_ddot(n, r_old, 1, sr_old, 1);
        cblas_dscal(n, beta, p, 1);
        cblas_daxpy(n, 1, r, 1, p, 1);
        cblas_dscal(n, beta, sp, 1);
        cblas_daxpy(n, 1, sr, 1, sp, 1);

        error = cblas_dnrm2(n, r, 1) / fnb;
        if (error < tol) {
            break;
        }
    }

    printf("%d, %e\n",i,error);
    free(tmp);
    free(atp);
    free(ap);
    free(sp);
    free(p);
    free(sr);
    free(r_old);
    free(sr_old);
    return 0;
}

