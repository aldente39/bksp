/*****
    This program is solving a linear system Ax = b.
    In this program, the algorithm to solve it
    is the BiCGSTAB(l) method.
*****/

#include "bksp.h"
#include <stdio.h>
#include <complex.h>
#include "bksp_internal.h"


int dbicgstabl (dspmat *A, double *b, int l,
                double tol, int max_iter, double *x) {
    ////////// Initialization //////////
    int i, j, m;
    int l2 = l + 1;
    double error = 0;
    double beta, *omega, gamma1, fnb;
    double alpha = 0;
    double gamma0 = 1;
    double **r, **p, **index;
    double *rs, *tmp, *base;
    double *sigma, *tau, *omega_d, *omega_dd, *omega_base, t_local;
    MKL_INT n = *A->row_size;
    index = (double **)malloc(l2 * 2 * sizeof(double *));
    r = index;
    p = &index[l2];
    base = (double *)malloc(n * (l2 * 2 + 2) * sizeof(double));
    for (i = 0; i < l2; i++) {
        r[i] = &base[n * i];
        p[i] = &base[n * (l2 + i)];
    }
    rs = &base[n * l2 * 2];
    tmp = &base[n * (l2 * 2 + 1)];
    omega_base = (double *)malloc(l2 * 4 * sizeof(double));
    omega = omega_base;
    omega[l] = 1;
    omega_d = &omega_base[l2];
    omega_dd = &omega_base[l2 * 2];
    sigma = &omega_base[l2 * 3];
    tau = (double *) malloc(l2 * l2 * sizeof(double));

    mkl_cspblas_dcsrgemv("n", A->row_size, A->value,
                            A->row, A->col, x, tmp);
    cblas_dscal(n, -1, tmp, 1);
    cblas_daxpy(n, 1, b, 1, tmp, 1);
    cblas_dcopy(n, tmp, 1, r[0], 1);
    cblas_dcopy(n, r[0], 1, p[0], 1);
    cblas_dcopy(n, r[0], 1, rs, 1);
    fnb = cblas_dnrm2(n, b, 1);

    ////////// Iteration //////////
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
                                    A->row, A->col, p[j], p[j + 1]);

            alpha = gamma1 / cblas_ddot(n, rs, 1, p[j + 1], 1);

            for (i = 0; i <= j; i++) {
                cblas_daxpy(n, -alpha, p[i + 1], 1, r[i], 1);
            }
            mkl_cspblas_dcsrgemv("n", A->row_size, A->value,
                                    A->row, A->col, r[j], r[j + 1]);

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

    ////////// Finalization //////////
    free(tau);
    free(omega_base);
    free(base);
    free(index);
    
    if (error >= tol) {
        return NOT_CONVERGED;
    }
    return m;
}


int zbicgstabl (zspmat *A, double _Complex *b, int l,
                double tol, int max_iter, double _Complex *x) {
    ////////// Initialization //////////
    int i, j, m;
    int l2 = l + 1;
    double error = 0;
    double one = 1;
    double fnb;
    double _Complex beta, *omega, gamma1;
    double _Complex alpha = 0 + 0 * _Complex_I;
    double _Complex gamma0 = 1 + 0 * _Complex_I;
    double _Complex **r, **p, **index;
    double _Complex *tmp, *base;
    double _Complex *rs;
    double _Complex *sigma, *tau, *omega_d, *omega_dd, *omega_base, t_local;
    MKL_INT n = *A->row_size;
    index = (double _Complex **)malloc(l2 * 2 * sizeof(double _Complex *));
    r = index;
    p = &index[l2];
    base = (double _Complex *)calloc(sizeof(double _Complex), n * (l2 * 2 + 2));
    for (i = 0; i < l2; i++) {
        r[i] = &base[n * i];
        p[i] = &base[n * (l2 + i)];
    }
    rs = &base[n * l2 * 2];
    tmp = &base[n * (l2 * 2 + 1)];
    omega_base = (double _Complex *)malloc(l2 * 4 * sizeof(double _Complex));
    omega = omega_base;
    omega[l] = 1;
    omega_d = &omega_base[l2];
    omega_dd = &omega_base[l2 * 2];
    sigma = &omega_base[l2 * 3];
    tau = (double _Complex *) malloc(l2 * l2 * sizeof(double _Complex));

    mkl_cspblas_zcsrgemv("n", A->row_size, A->value,
                            A->row, A->col, x, tmp);
    cblas_zdscal(n, -1, tmp, 1);
    cblas_zaxpy(n, &one, b, 1, tmp, 1);
    cblas_zcopy(n, tmp, 1, r[0], 1);
    cblas_zcopy(n, r[0], 1, p[0], 1);
    cblas_zcopy(n, r[0], 1, rs, 1);
    fnb = cblas_dznrm2(n, b, 1);

    // ********** Outer teration **********
    for (m = 0; m < max_iter; m++) {
        // ********** BiCG Part **********
        gamma0 = -omega[l] * gamma0;

        for (j = 0; j < l; j++) {
            cblas_zdotc_sub(n, rs, 1, r[j], 1, &gamma1);
            beta = -alpha * gamma1 / gamma0;
            gamma0 = gamma1;

            for (i = 0; i <= j; i++) {
                cblas_zscal(n, &beta, p[i], 1);
                cblas_zaxpy(n, &one, r[i], 1, p[i], 1);
            }
            mkl_cspblas_zcsrgemv("n", A->row_size, A->value,
                                 A->row, A->col, p[j], p[j + 1]);

            cblas_zdotc_sub(n, rs, 1, p[j + 1], 1, &tmp[0]);
            alpha = gamma1 / tmp[0];

            tmp[0] = -alpha;
            for (i = 0; i <= j; i++) {
                cblas_zaxpy(n, &tmp[0], p[i + 1], 1, r[i], 1);
            }
            mkl_cspblas_zcsrgemv("n", A->row_size, A->value,
                                 A->row, A->col, r[j], r[j + 1]);

            cblas_zaxpy(n, &alpha, p[0], 1, x, 1);
        }


        // ********** MR Part **********
        for (j = 1; j <= l; j++) {
            for (i = 1; i < j; i++) {
                cblas_zdotc_sub(n, r[i], 1, r[j], 1, &tmp[0]);
                tau[j * l + i] = tmp[0] / sigma[i];
                tmp[0] = -tau[j * l + i];
                cblas_zaxpy(n, &tmp[0], r[i], 1, r[j], 1);
            }
            cblas_zdotc_sub(n, r[j], 1, r[j], 1, &sigma[j]);
            cblas_zdotc_sub(n, r[j], 1, r[0], 1, &tmp[0]);
            omega_d[j] = tmp[0] / sigma[j];
        }

        omega[l] = omega_d[l];

        for (j = l - 1; j > 0; j--) {
            t_local = 0;
            for (i = j + 1; i <= l; i++) {
                t_local += tau[i * l + j] * omega[i];
            }
            omega[j] = omega_d[j] - t_local;
        }

        for (j = 1; j < l; j++) {
            t_local = 0;
            for (i = j + 1; i < l; i++) {
                t_local += tau[i * l + j] * omega[i + 1];
            }
            omega_dd[j] = omega[j + 1] + t_local;
        }

        // ********** Update **********
        tmp[0] = -omega[l];
        tmp[1] = -omega_d[l];
        cblas_zaxpy(n, &omega[1], r[0], 1, x, 1);
        cblas_zaxpy(n, &tmp[1], r[l], 1, r[0], 1);
        cblas_zaxpy(n, &tmp[0], p[l], 1, p[0], 1);
        for (i = 1; i < l; i++) {
            tmp[0] = -omega[i];
            tmp[1] = -omega_d[i];
            cblas_zaxpy(n, &tmp[0], p[i], 1, p[0], 1);
            cblas_zaxpy(n, &tmp[1], r[i], 1, r[0], 1);
            cblas_zaxpy(n, &omega_dd[i], r[i], 1, x, 1);
        }

        error = cblas_dznrm2(n, r[0], 1) / fnb;
        if (error < tol) {
            break;
        }
    }

    ////////// Finalization //////////
    free(tau);
    free(omega_base);
    free(base);
    free(index);
    
    if (error >= tol) {
        return NOT_CONVERGED;
    }
    return m;
}
