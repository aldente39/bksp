#include <mkl.h>
#include "bksp.h"
#include <stdio.h>

int dbicgstab(dspmat *mat, double *b, double tol,
                                       int max_iter, double *x) {
    ////////// Initialization //////////
    int i;
    double error = 0;
    double alpha, beta, omega, rho, rho_new, norm_b;
    double *r, *rs, *s, *p, *ap, *as, *tmp, *base;
    MKL_INT n = *mat->row_size;
    base = (double *)malloc(n * 7 * sizeof(double));
    r = base;
    rs = &base[n];
    s = &base[n * 2];
    p = &base[n * 3];
    ap = &base[n * 4];
    as = &base[n * 5];
    tmp = &base[n * 6];

    mkl_cspblas_dcsrgemv("n", mat->row_size, mat->value,
                   mat->I, mat->J, x, tmp);
    cblas_dscal(n, -1, tmp, 1);
    cblas_daxpy(n, 1, b, 1, tmp, 1);
    cblas_dcopy(n, tmp, 1, r, 1);
    cblas_dcopy(n, r, 1, p, 1);
    cblas_dcopy(n, r, 1, rs, 1);
    norm_b = cblas_dnrm2(n, b, 1);
    rho = cblas_ddot(n, rs, 1, r, 1);

    ////////// Iteration //////////
    for (i = 0; i < max_iter; i++) {
        mkl_cspblas_dcsrgemv("n", mat->row_size, mat->value,
                     mat->I, mat->J, p, ap);
        alpha = rho / cblas_ddot(n, rs, 1, ap, 1);

        cblas_dcopy(n, r, 1, s, 1);
        cblas_daxpy(n, -alpha, ap, 1, s, 1);

        mkl_cspblas_dcsrgemv("n", mat->row_size, mat->value,
                    mat->I, mat->J, s, as);
        omega = cblas_ddot(n, as, 1, s, 1) / cblas_ddot(n, as, 1, as, 1);

        cblas_daxpy(n, alpha, p, 1, x, 1);
        cblas_daxpy(n, omega, s, 1, x, 1);

        cblas_dcopy(n, s, 1, r, 1);
        cblas_daxpy(n, -omega, as, 1, r, 1);

        rho_new = cblas_ddot(n, rs, 1, r, 1);
        beta = alpha / omega * rho_new / rho;
        rho = rho_new;

        cblas_daxpy(n, -omega, ap, 1, p, 1);
        cblas_dscal(n, beta, p, 1);
        cblas_daxpy(n, 1, r, 1, p, 1);

        error = cblas_dnrm2(n, r, 1) / norm_b;
        if (error < tol) {
            break;
        }
    }

    ////////// Finalization //////////
    free(base);
    
    if (error >= tol) {
        return NOT_CONVERGED;
    }
    return i;
}

