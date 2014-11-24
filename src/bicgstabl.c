/*****
    This program is solving a linear system Ax = b.
    In this program, the algorithm to solve it
    is the BiCGSTAB(l) method.
*****/

#include <mkl.h>
#include "bksp.h"
#include <stdio.h>

int dbicgstabl (dspmat *A, double *b, int l, double tol,
                                       int max_iter, double *x) {
    int i, j, m, info;
    int l2 = l + 1;
    double error = 0;
    double beta, *omega, gamma1, fnb;
    double alpha = 0;
    double gamma0 = 1;
    double **r, *rs, **p;
    double *sigma, *tau, *omega_d, *omega_dd, t_local;
    MKL_INT n = *A->row_size;
    r = malloc(l2 * sizeof(double));
    p = malloc(l2 * sizeof(double));
    for (i = 0; i < l2; i++) {
        r[i] = (double *) malloc(n * sizeof(double));
        p[i] = (double *) malloc(n * sizeof(double));
    }
    rs = (double *) malloc(n * sizeof(double));
    omega = (double *) malloc(l2 * sizeof(double));
    omega[l] = 1;
    omega_d = (double *) malloc(l2 * sizeof(double));
    omega_dd = (double *) malloc(l2 * sizeof(double));
    sigma = (double *) malloc(l2 * sizeof(double));
    tau = (double *) malloc(l2 * l2 * sizeof(double));

    double *tmp = (double *) malloc(n * sizeof(double));
    mkl_cspblas_dcsrgemv("n", A->row_size, A->value,
                            A->I, A->J, x, tmp);
    cblas_dscal(n, -1, tmp, 1);
    cblas_daxpy(n, 1, b, 1, tmp, 1);
    cblas_dcopy(n, tmp, 1, r[0], 1);
    cblas_dcopy(n, r[0], 1, p[0], 1);
    cblas_dcopy(n, r[0], 1, rs, 1);
    fnb = cblas_dnrm2(n, b, 1);

    for (m = 0; m < max_iter; m++) {
        // BiCG Part

        gamma0 = -omega[l] * gamma0;

        for (j = 0; j < l; j++) {
            gamma1 = cblas_ddot(n, rs, 1, r[j], 1);
            beta = -alpha * gamma1 / gamma0;
            gamma0 = gamma1;

            for (i = 0; i <= j; i++) {
                cblas_dscal(n, beta, p[i], 1);
                cblas_daxpy(n, 1, r[i], 1, p[i], 1);
            }
            mkl_cspblas_dcsrgemv("n", A->row_size, A->value,
                                    A->I, A->J, p[j], p[j + 1]);

            alpha = gamma1 / cblas_ddot(n, rs, 1, p[j + 1], 1);

            for (i = 0; i <= j; i++) {
                cblas_daxpy(n, -alpha, p[i + 1], 1, r[i], 1);
            }
            mkl_cspblas_dcsrgemv("n", A->row_size, A->value,
                                    A->I, A->J, r[j], r[j + 1]);

            cblas_daxpy(n, alpha, p[0], 1, x, 1);
        }

        // MR Part

        for (j = 1; j <= l; j++) {
            for (i = 1; i < j; i++) {
                tau[i * l + j] = cblas_ddot(n, r[j], 1, r[i], 1) / sigma[i];
                cblas_daxpy(n, -tau[i * l + j], r[i], 1, r[j], 1);
            }
            sigma[j] = cblas_ddot(n, r[j], 1, r[j], 1);
            omega_d[j] = cblas_ddot(n, r[0], 1, r[j], 1) / sigma[j];
        }

        omega[l] = omega_d[l];

        for (j = l - 1; j > 0; j--) {
            t_local = 0;
            for (i = j + 1; i <= l; i++) {
                t_local += tau[j * l + i] * omega[i];
            }
            omega[j] = omega_d[j] - t_local;
        }

        for (j = 1; j < l; j++) {
            t_local = 0;
            for (i = j + 1; i < l; i++) {
                t_local += tau[j * l + i] * omega[i + 1];
            }
            omega_dd[j] = omega[j + 1] + t_local;
        }

        // Update

        cblas_daxpy(n, omega[1], r[0], 1, x, 1);
        cblas_daxpy(n, -omega_d[l], r[l], 1, r[0], 1);
        cblas_daxpy(n, -omega[l], p[l], 1, p[0], 1);
        for (i = 1; i < l; i++) {
            cblas_daxpy(n, -omega[i], p[i], 1, p[0], 1);
            cblas_daxpy(n, -omega_d[i], r[i], 1, r[0], 1);
            cblas_daxpy(n, omega_dd[i], r[i], 1, x, 1);
        }

        error = cblas_dnrm2(n, r[0], 1) / fnb;
        if (error < tol) {
            break;
        }
    }
    printf("%d, %e\n", m, error);

    free(tmp);
    free(tau);
    free(sigma);
    free(omega_dd);
    free(omega_d);
    free(omega);
    free(rs);
    for (i = 1; i < l2; i++) {
        free(p[i]);
        free(r[i]);
    }
    free(p[0]);
    return 0;
}

