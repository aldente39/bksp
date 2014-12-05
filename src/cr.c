#include <stdio.h>
#include <mkl.h>
#include "bksp.h"

int dcr (dspmat *mat, double *b,
            double tol, int max_iter, double *x) {

    if (strcmp(mat->type, "symmetric") != 0) {
        return INVALID_MATRIX_TYPE;
    }

    ////////// Initialization //////////
    int i;
    double error = 0;
    double alpha, beta, norm_b, r_dot, r_dot_old;
    double *r, *p, *ap, *tmp;
    MKL_INT n = *mat->row_size;
    r = (double *) malloc(n * sizeof(double));
    p = (double *) malloc(n * sizeof(double));
    ap = (double *) malloc(n * sizeof(double));
    tmp = (double *) malloc(n * sizeof(double));

    mkl_cspblas_dcsrsymv("l", mat->row_size, mat->value,
                   mat->I, mat->J, x, tmp);
    cblas_dscal(n, -1, tmp, 1);
    cblas_daxpy(n, 1, b, 1, tmp, 1);
    cblas_dcopy(n, tmp, 1, r, 1);
    cblas_dcopy(n, r, 1, p, 1);
    mkl_cspblas_dcsrsymv("l", mat->row_size, mat->value,
                   mat->I, mat->J, r, tmp);
    cblas_dcopy(n, tmp, 1, ap, 1);
    r_dot_old = cblas_ddot(n, r, 1, tmp, 1);
    norm_b = cblas_dnrm2(n, b, 1);

    ////////// Iteration //////////
    for (i = 1; i < max_iter; i++) {
        // Compute alpha
        alpha = r_dot_old / cblas_ddot(n, ap, 1, ap, 1);

        // Compute the approximate solution x.
        cblas_daxpy(n, alpha, p, 1, x, 1);

        // Compute the residual r.
        cblas_daxpy(n, -alpha, ap, 1, r, 1);

        // Compute A * r.
        mkl_cspblas_dcsrsymv("l", mat->row_size, mat->value,
                     mat->I, mat->J, r, tmp);

        // Compute beta.
        r_dot = cblas_ddot(n, r, 1, tmp, 1);
        beta = r_dot / r_dot_old;
        r_dot_old = r_dot;

        // Compute p.
        cblas_dscal(n, beta, p, 1);
        cblas_daxpy(n, 1, r, 1, p, 1);

        // Compute A * p;
        cblas_dscal(n, beta, ap, 1);
        cblas_daxpy(n, 1, tmp, 1, ap, 1);

        error = cblas_dnrm2(n, r, 1) / norm_b;
        if (error < tol) {
            break;
        }
    }

    ////////// Finalization //////////
    free(tmp);
    free(ap);
    free(p);
    free(r);
    
    if (error >= tol) {
        return NOT_CONVERGED;
    }
    return i;
}

