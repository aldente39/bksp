#include <stdio.h>
#include <mkl.h>
#include "bksp.h"

int dbicg (dspmat *mat, double *b,
            double tol, int max_iter, double *x) {
    ////////// Initialization //////////
    int i;
    double error = 0;
    double alpha, beta, norm_b, r_dot, r_dot_old;
    double *r, *sr, *p, *sp, *ap, *atp;
    MKL_INT n = *mat->row_size;
    r = (double *) malloc(n * sizeof(double));
    sr = (double *) malloc(n * sizeof(double));
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
    r_dot_old = cblas_ddot(n, r, 1, sr, 1);
    norm_b = cblas_dnrm2(n, b, 1);

    ////////// Iteration //////////
    for (i = 0; i < max_iter; i++) {
        // Compute A * p.
        mkl_cspblas_dcsrgemv("n", mat->row_size, mat->value,
                     mat->I, mat->J, p, ap);

        // Compute A^{T} * sp.
        mkl_cspblas_dcsrgemv("t", mat->row_size, mat->value,
                     mat->I, mat->J, sp, atp);

        // Compute alpha.
        alpha = r_dot_old / cblas_ddot(n, ap, 1, sp, 1);

        // Compute the approximate solution x.
        cblas_daxpy(n, alpha, p, 1, x, 1);

        // Compute the residual r.
        cblas_daxpy(n, -alpha, ap, 1, r, 1);

        // Compute the shadow residual sr.
        cblas_daxpy(n, -alpha, atp, 1, sr, 1);

        // Compute beta.
        r_dot = cblas_ddot(n, r, 1, sr, 1);
        beta = r_dot / r_dot_old;
        r_dot_old = r_dot;

        // Compute p.
        cblas_dscal(n, beta, p, 1);
        cblas_daxpy(n, 1, r, 1, p, 1);

        // Compute shadow p.
        cblas_dscal(n, beta, sp, 1);
        cblas_daxpy(n, 1, sr, 1, sp, 1);

        error = cblas_dnrm2(n, r, 1) / norm_b;
        if (error < tol) {
            break;
        }
    }

    ////////// Finalization //////////
    printf("%d, %e\n",i,error);
    free(tmp);
    free(atp);
    free(ap);
    free(sp);
    free(p);
    free(sr);
    free(r);
    
    if (error >= tol) {
        return -1;
    }
    return 0;
}

