#include <stdio.h>
#include <string.h>
#include "bksp_internal.h"
#include "bksp.h"

int bksp_dcg (dspmat *mat, double *b,
            double tol, int max_iter, double *x) {

    if (strcmp(mat->type, "symmetric") != 0) {
        return INVALID_MATRIX_TYPE;
    }

    ////////// Initialization //////////
    int i;
    double error = 0;
    double alpha, beta, fnb, r_dot, r_dot_old;
    double *r, *p, *tmp;
    MKL_INT n = *mat->row_size;
    r = (double *) malloc(n * sizeof(double));
    p = (double *) malloc(n * sizeof(double));
    tmp = (double *) malloc(n * sizeof(double));

    mkl_cspblas_dcsrsymv("l", mat->row_size, mat->value,
                   mat->row, mat->col, x, tmp);
    cblas_dscal(n, -1, tmp, 1);
    cblas_daxpy(n, 1, b, 1, tmp, 1);
    cblas_dcopy(n, tmp, 1, r, 1);
    cblas_dcopy(n, r, 1, p, 1);
    r_dot_old = cblas_ddot(n, r, 1, r, 1);
    fnb = cblas_dnrm2(n, b, 1);

    ////////// Iteration //////////
    for (i = 0; i < max_iter; i++) {
        // Compute A * p.
        mkl_cspblas_dcsrsymv("l", mat->row_size, mat->value,
                     mat->row, mat->col, p, tmp);

        // Compute alpha
        alpha = r_dot_old / cblas_ddot(n, tmp, 1, p, 1);

        // Compute the approximate solution x.
        cblas_daxpy(n, alpha, p, 1, x, 1);

        // Compute the residual r.
        cblas_daxpy(n, -alpha, tmp, 1, r, 1);

        // Compute beta.
        r_dot = cblas_ddot(n, r, 1, r, 1);
        beta = r_dot / r_dot_old;
        r_dot_old = r_dot;

        // Compute p.
        cblas_dscal(n, beta, p, 1);
        cblas_daxpy(n, 1, r, 1, p, 1);

        error = cblas_dnrm2(n, r, 1) / fnb;
        if (error < tol) {
            break;
        }
    }

    ////////// Finalization //////////
    free(tmp);
    free(p);
    free(r);
    
    if (error >= tol) {
        return NOT_CONVERGED;
    }
    return i;
}

