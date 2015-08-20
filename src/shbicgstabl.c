#include <stdio.h>
#include <math.h>
#include "bksp_internal.h"
#include "bksp.h"


int dshbicgstabl (dspmat *mat, double *b, int l, double *sigma,
                  int sigma_size, double tol, int max_iter, double *x) {
    ////////// Initialization //////////
    int iter, i, j, k;
    int l2 = l + 1;
    double error = 0;
    double alpha, beta, alpha_old, rho, rho_old, norm_b;
    double **r, *rs, **p;
    double *s, *tau, *omega, *omega_d, *omega_dd, t_local;
    MKL_INT n = *mat->row_size;
    r = malloc(l2 * sizeof(double));
    p = malloc(l2 * sizeof(double));
    for (i = 0; i < l + 1; i++) {
        r[i] = (double *) malloc(n * sizeof(double));
        p[i] = (double *) malloc(n * sizeof(double));
    }
    rs = (double *) malloc(n * sizeof(double));
    omega = (double *) calloc(l, sizeof(double));
    omega[l - 1] = 1;
    omega_d = (double *) calloc(l, sizeof(double));
    omega_dd = (double *) calloc(l, sizeof(double));
    s = (double *) calloc(l, sizeof(double));
    tau = (double *) calloc(l * l, sizeof(double));

    double *tmp = (double *) malloc(n * sizeof(double));
    mkl_cspblas_dcsrgemv("n", mat->row_size, mat->value,
                         mat->row, mat->col, x, tmp);
    cblas_dscal(n, -1, tmp, 1);
    cblas_daxpy(n, 1, b, 1, tmp, 1);
    cblas_dcopy(n, tmp, 1, r[0], 1);
    cblas_dcopy(n, r[0], 1, p[0], 1);
    cblas_dcopy(n, r[0], 1, rs, 1);
    norm_b = cblas_dnrm2(n, b, 1);
    alpha = 0.0;
    alpha_old = 1.0;
    rho_old = 1.0;

    // For shifted systems
    double *alpha_shift = (double *)malloc(sigma_size * sizeof(double));
    double *beta_shift = (double *)malloc(sigma_size * sizeof(double));
    double *pi_shift = (double *)malloc(sigma_size * sizeof(double));
    double *pi_shift_new = (double *)malloc(sigma_size * sizeof(double));
    double *pi_shift_old = (double *)malloc(sigma_size * sizeof(double));
    double ***p_shift = (double ***)malloc(sigma_size * sizeof(double ***));
    double **p_shift_ind = (double **)malloc(sigma_size * l2 * sizeof(double **));
    double **x_shift = (double **)malloc(sigma_size * sizeof(double *));
    double *p_shift_base = (double *)calloc(n * l2 * sigma_size, sizeof(double));
    double *theta_shift = (double *) malloc(sigma_size * sizeof(double));
    double *omega_shift, *omega_shift_d, *omega_shift_dd;
    double psi_shift, e_shift;
    omega_shift = (double *) malloc(l * sizeof(double));
    omega_shift_d = (double *) malloc(l * sizeof(double));
    omega_shift_dd = (double *) malloc(l * sizeof(double));
    for (i = 0; i < sigma_size; i++) {
        p_shift[i] = &p_shift_ind[i * l2];
        x_shift[i] = &x[(i + 1) * n];
        pi_shift_old[i] = 1;
        pi_shift[i] = 1;
        theta_shift[i] = 1;
    }
    for (i = 0; i < sigma_size * l2; i++) {
        p_shift_ind[i] = &p_shift_base[i * n];
    }
    double **M = (double **) malloc(sigma_size * sizeof(double));
    double *M_base = (double *) calloc(sigma_size * l * l, sizeof(double *));
    for (k = 0; k < sigma_size; k++) {
        M[k] = &M_base[k * l * l];
    }
    for (k = 0; k < sigma_size; k++) {
        for (j = 0; j < l; j++) {
            for (i = 0; i <= j; i++) {
                M[k][j * l + i] = pow(sigma[k], j - i) * comb(j, i);
            }
        }
    }

    ////////// Iteration //////////
    for (iter = 0; iter < max_iter; iter++) {

        ///// BiCG Part

        rho_old = - omega[l - 1] * rho_old;
        for (j = 0; j < l; j++) {
            // Compute beta.
            rho = cblas_ddot(n, r[j], 1, rs, 1);
            beta = alpha * rho / rho_old;

            // Compute p_{j, i}
            for (i = 0; i <= j; i++) {
                cblas_dscal(n, -beta, p[i], 1);
                cblas_daxpy(n, 1, r[i], 1, p[i], 1);
            }

            // Compute p_{j, j+1}.
            mkl_cspblas_dcsrgemv("n", mat->row_size, mat->value,
                                 mat->row, mat->col, p[j], p[j + 1]);

            // Compute alpha.
            alpha = rho / cblas_ddot(n, rs, 1, p[j + 1], 1);

            // Compute x.
            cblas_daxpy(n, alpha, p[0], 1, x, 1);

            // Shifted systems.
            for (k = 0; k < sigma_size; k++) {
                // Compute pi_shift_new.
                pi_shift_new[k] = (1.0 + alpha * sigma[k]) * pi_shift[k]
                    + alpha * beta / alpha_old * (pi_shift_old[k] - pi_shift[k]);
            
                // Compute beta_shift.
                beta_shift[k] = (pi_shift_old[k] / pi_shift[k]) *
                    (pi_shift_old[k] / pi_shift[k]) * beta;
            
                // Compute alpha_shift.
                alpha_shift[k] = pi_shift[k] / pi_shift_new[k] * alpha;

                // Compute p_shift.
                for (i = 0; i <= j; i++) {
                    cblas_dscal(n, -beta_shift[k], p_shift[k][i], 1);
                    cblas_daxpy(n, 1.0 / (pi_shift[k] * theta_shift[k]),
                                r[i], 1, p_shift[k][i], 1);
                }
                cblas_dcopy(n, r[j], 1, p_shift[k][j + 1], 1);
                cblas_dscal(n, 1.0 / pi_shift[k], p_shift[k][j + 1], 1);

                // Compute x_shift.
                cblas_daxpy(n, alpha_shift[k], p_shift[k][0], 1, x_shift[k], 1);
            }
        
            // Compute r_{j+1, i}.
            for (i = 0; i <= j; i++) {
                cblas_daxpy(n, -alpha, p[i + 1], 1, r[i], 1);
            }

            // Compute r_{j+1, j+1}.
            mkl_cspblas_dcsrgemv("n", mat->row_size, mat->value,
                                 mat->row, mat->col, r[j], r[j + 1]);

            // Compute p_shift_{j, j+1} of shifted systems.
            for (k = 0; k < sigma_size; k++) {
                cblas_daxpy(n, -1.0 / pi_shift_new[k],
                            r[j], 1, p_shift[k][j + 1], 1);
                cblas_dscal(n, 1.0 / (theta_shift[k] * alpha_shift[k]),
                            p_shift[k][j + 1], 1);
                cblas_daxpy(n, - sigma[k], p_shift[k][j], 1,
                            p_shift[k][j + 1], 1);
            }

            // Update scalars.
            alpha_old = alpha;
            rho_old = rho;
            for (k = 0; k < sigma_size; k++) {
                pi_shift_old[k] = pi_shift[k];
                pi_shift[k] = pi_shift_new[k];
            }
        }

        ///// MR Part

        for (j = 0; j < l; j++) {
            for (i = 0; i < j; i++) {
                tau[i * l + j] = cblas_ddot(n, r[j + 1], 1, r[i + 1], 1) / s[i];
                cblas_daxpy(n, -tau[i * l + j], r[i + 1], 1, r[j + 1], 1);
            }
            s[j] = cblas_ddot(n, r[j + 1], 1, r[j + 1], 1);
            omega_d[j] = cblas_ddot(n, r[0], 1, r[j + 1], 1) / s[j];
        }

        omega[l - 1] = omega_d[l - 1];

        for (j = l - 2; j >= 0; j--) {
            t_local = 0;
            for (i = j + 1; i < l; i++) {
                t_local += tau[j * l + i] * omega[i];
            }
            omega[j] = omega_d[j] - t_local;
        }

        for (j = 0; j < l - 1; j++) {
            t_local = 0;
            for (i = j + 1; i < l - 1; i++) {
                t_local += tau[j * l + i] * omega[i + 1];
            }
            omega_dd[j] = omega[j + 1] + t_local;
        }

        // Update the seed system except the residual.
        cblas_daxpy(n, omega[0], r[0], 1, x, 1);
        cblas_daxpy(n, -omega[l - 1], p[l], 1, p[0], 1);
        for (i = 1; i < l; i++) {
            cblas_daxpy(n, -omega[i - 1], p[i], 1, p[0], 1);
            cblas_daxpy(n, omega_dd[i - 1], r[i], 1, x, 1);
        }

        // Shifted systems.
        for (k = 0; k < sigma_size; k++) {
            dhorner(omega, l, sigma[k], tmp);
            psi_shift = tmp[0];
            for (i = 0; i < l; i++) {
                omega_shift[i] = tmp[i + 1];
            }

            e_shift = theta_shift[k] * pi_shift[k];
            theta_shift[k] = theta_shift[k] * psi_shift;

            for (j = 0; j < l; j++) {
                tmp[0] = 0;
                for (i = j; i < l; i++) {
                    tmp[0] += M[k][i * l + j] * omega_shift[i];
                }
                omega_shift_d[j] = tmp[0];
            }

            for (j = 0; j < l - 1; j++) {
                tmp[0] = 0;
                for (i = j + 1; i < l - 1; i++) {
                    tmp[0] += tau[j * l + i] * omega_shift_d[i + 1];
                }
                omega_shift_dd[j] = omega_shift_d[j + 1] + tmp[0];
            }

            // Update shifted systems.
            cblas_daxpy(n, omega_shift_d[0] / e_shift, r[0], 1, x_shift[k], 1);
            cblas_daxpy(n, -omega[l - 1], p_shift[k][l], 1, p_shift[k][0], 1);
            for (i = 1; i < l; i++) {
                cblas_daxpy(n, -omega[i - 1], p_shift[k][i], 1, p_shift[k][0], 1);
                cblas_daxpy(n, omega_shift_dd[i - 1] / e_shift,
                            r[i], 1, x_shift[k], 1);
            }
            cblas_dscal(n, 1.0 / psi_shift, p_shift[k][0], 1);
        }

        // Update the residual of the seed system.
        for (i = 1; i <= l; i++) {
            cblas_daxpy(n, -omega_d[i - 1], r[i], 1, r[0], 1);
        }

        error = cblas_dnrm2(n, r[0], 1) / norm_b;
        if (error < tol) {
            break;
        }
    }

    ////////// Finalization //////////
    free(tmp);
    free(tau);
    free(s);
    free(omega_dd);
    free(omega_d);
    free(omega);
    free(rs);
    for (i = 0; i <= l; i++) {
        free(p[i]);
        free(r[i]);
    }
    free(p);
    free(r);
    free(alpha_shift);
    free(beta_shift);
    free(pi_shift);
    free(pi_shift_old);
    free(pi_shift_new);
    free(omega_shift_dd);
    free(omega_shift_d);
    free(omega_shift);
    free(theta_shift);
    free(p_shift);
    free(p_shift_ind);
    free(p_shift_base);
    free(x_shift);
    free(M_base);
    free(M);
    
    if (error >= tol) {
        return NOT_CONVERGED;
    }

    return iter;
}


int zshbicgstabl (zspmat *mat, double _Complex *b, int l, double _Complex *sigma,
                  int sigma_size, double tol, int max_iter, double _Complex *x) {
    ////////// Initialization //////////
    double _Complex mvalue;
    double _Complex one, mone, zero;
    one = 1;
    mone = -1;
    zero = 0;

    int iter, i, j, k;
    int l2 = l + 1;
    double error = 0;
    double _Complex alpha, beta, alpha_old, rho, rho_old, norm_b;
    double _Complex  **r, *rs, **p, *base, *tmp;
    double _Complex *s, *tau, *omega, *omega_d, *omega_dd, *omega_base, t_local;
    MKL_INT n = *mat->row_size;
    r = malloc(l2 * sizeof(double _Complex *));
    p = malloc(l2 * sizeof(double _Complex *));
    base = (double _Complex *)calloc(sizeof(double _Complex),
                                     n * (l2 * 2 + 2));
    for (i = 0; i < l2; i++) {
        r[i] = &base[n * i];
        p[i] = &base[n * (l2 + i)];
    }
    rs = &base[n * l2 * 2];
    tmp = &base[n * (l2 * 2 + 1)];
    omega_base = (double _Complex*)calloc(l * 4, sizeof(double _Complex));
    omega = omega_base;
    omega[l - 1] = 1;
    omega_d = &omega_base[l];
    omega_dd = &omega_base[l * 2];
    s = &omega_base[l * 3];
    tau = (double _Complex*)calloc(l * l, sizeof(double _Complex));

    mkl_cspblas_zcsrgemv("n", mat->row_size, mat->value,
                         mat->row, mat->col, x, tmp);
    cblas_zdscal(n, -1, tmp, 1);
    cblas_zaxpy(n, &one, b, 1, tmp, 1);
    cblas_zcopy(n, tmp, 1, r[0], 1);
    cblas_zcopy(n, r[0], 1, p[0], 1);
    cblas_zcopy(n, r[0], 1, rs, 1);
    norm_b = cblas_dznrm2(n, b, 1);
    alpha = 0.0;
    alpha_old = 1.0;
    rho_old = 1.0;

    // For shifted systems
    double _Complex *alpha_shift, *beta_shift, *small_mat_shift_base,
        *pi_shift, *pi_shift_new, *pi_shift_old;
    small_mat_shift_base = (double _Complex *)malloc(sizeof(double _Complex) *
                                                     sigma_size * 5);
    alpha_shift = small_mat_shift_base;
    beta_shift = &small_mat_shift_base[sigma_size];
    pi_shift = &small_mat_shift_base[sigma_size * 2];
    pi_shift_new = &small_mat_shift_base[sigma_size * 3];
    pi_shift_old = &small_mat_shift_base[sigma_size * 4];
    double _Complex ***p_shift, **p_shift_ind, *p_shift_base,
        **x_shift, *theta_shift;
    p_shift = (double _Complex ***)malloc(sigma_size *
                                          sizeof(double _Complex ***));
    p_shift_ind = (double _Complex **)malloc(sigma_size * l2 *
                                             sizeof(double _Complex **));
    x_shift = (double _Complex **)malloc(sigma_size * sizeof(double _Complex *));
    p_shift_base = (double _Complex *)calloc(n * l2 * sigma_size,
                                             sizeof(double _Complex));
    theta_shift = (double _Complex *)malloc(sigma_size * sizeof(double _Complex));
    double _Complex *omega_shift, *omega_shift_d,
        *omega_shift_dd, *omega_shift_base;
    double _Complex psi_shift, e_shift;
    omega_shift_base = (double _Complex *)malloc(3 * l * sizeof(double _Complex));
    omega_shift = omega_shift_base;
    omega_shift_d = &omega_shift_base[l];
    omega_shift_dd = &omega_shift_base[l * 2];
    for (i = 0; i < sigma_size; i++) {
        p_shift[i] = &p_shift_ind[i * l2];
        x_shift[i] = &x[(i + 1) * n];
        pi_shift_old[i] = 1;
        pi_shift[i] = 1;
        theta_shift[i] = 1;
    }
    for (i = 0; i < sigma_size * l2; i++) {
        p_shift_ind[i] = &p_shift_base[i * n];
    }
    double _Complex **M = (double _Complex **)malloc(sigma_size *
                                                     sizeof(double _Complex *));
    double _Complex *M_base = (double _Complex *)calloc(sigma_size * l * l,
                                                        sizeof(double _Complex));
    for (k = 0; k < sigma_size; k++) {
        M[k] = &M_base[k * l * l];
    }
    for (k = 0; k < sigma_size; k++) {
        for (j = 0; j < l; j++) {
            for (i = 0; i <= j; i++) {
                M[k][j * l + i] = cpow(sigma[k], j - i) * comb(j, i);
            }
        }
    }

    ////////// Iteration //////////
    for (iter = 0; iter < max_iter; iter++) {

        ///// BiCG Part

        rho_old = - omega[l - 1] * rho_old;
        for (j = 0; j < l; j++) {
            // Compute beta.
            cblas_zdotc_sub(n, rs, 1, r[j], 1, &rho);
            beta = alpha * rho / rho_old;

            // Compute p_{j, i}
            for (i = 0; i <= j; i++) {
                mvalue = -beta;
                cblas_zscal(n, &mvalue, p[i], 1);
                cblas_zaxpy(n, &one, r[i], 1, p[i], 1);
            }

            // Compute p_{j, j+1}.
            mkl_cspblas_zcsrgemv("n", mat->row_size, mat->value,
                                 mat->row, mat->col, p[j], p[j + 1]);

            // Compute alpha.
            cblas_zdotc_sub(n, rs, 1, p[j + 1], 1, &mvalue);
            alpha = rho / mvalue;

            // Compute x.
            cblas_zaxpy(n, &alpha, p[0], 1, x, 1);

            // Shifted systems.
            for (k = 0; k < sigma_size; k++) {
                // Compute pi_shift_new.
                pi_shift_new[k] = (1.0 + alpha * sigma[k]) * pi_shift[k]
                    + alpha * beta / alpha_old * (pi_shift_old[k] - pi_shift[k]);
            
                // Compute beta_shift.
                beta_shift[k] = (pi_shift_old[k] / pi_shift[k]) *
                    (pi_shift_old[k] / pi_shift[k]) * beta;
            
                // Compute alpha_shift.
                alpha_shift[k] = pi_shift[k] / pi_shift_new[k] * alpha;

                // Compute p_shift.
                for (i = 0; i <= j; i++) {
                    mvalue = -beta_shift[k];
                    cblas_zscal(n, &mvalue, p_shift[k][i], 1);
                    mvalue = 1.0 / (pi_shift[k] * theta_shift[k]);
                    cblas_zaxpy(n, &mvalue,
                                r[i], 1, p_shift[k][i], 1);
                }
                cblas_zcopy(n, r[j], 1, p_shift[k][j + 1], 1);
                mvalue =  1.0 / pi_shift[k];
                cblas_zscal(n, &mvalue, p_shift[k][j + 1], 1);

                // Compute x_shift.
                cblas_zaxpy(n, &alpha_shift[k], p_shift[k][0], 1, x_shift[k], 1);
            }
        
            // Compute r_{j+1, i}.
            mvalue = -alpha;
            for (i = 0; i <= j; i++) {
                cblas_zaxpy(n, &mvalue, p[i + 1], 1, r[i], 1);
            }

            // Compute r_{j+1, j+1}.
            mkl_cspblas_zcsrgemv("n", mat->row_size, mat->value,
                                 mat->row, mat->col, r[j], r[j + 1]);

            // Compute p_shift_{j, j+1} of shifted systems.
            for (k = 0; k < sigma_size; k++) {
                mvalue =  -1.0 / pi_shift_new[k];
                cblas_zaxpy(n, &mvalue, r[j], 1, p_shift[k][j + 1], 1);
                mvalue =  1.0 / (theta_shift[k] * alpha_shift[k]);
                cblas_zscal(n, &mvalue, p_shift[k][j + 1], 1);
                mvalue =  -sigma[k];
                cblas_zaxpy(n, &mvalue, p_shift[k][j], 1,
                            p_shift[k][j + 1], 1);
            }

            // Update scalars.
            alpha_old = alpha;
            rho_old = rho;
            for (k = 0; k < sigma_size; k++) {
                pi_shift_old[k] = pi_shift[k];
                pi_shift[k] = pi_shift_new[k];
            }
        }


        ///// MR Part

        for (j = 0; j < l; j++) {
            for (i = 0; i < j; i++) {
                cblas_zdotc_sub(n, r[i + 1], 1, r[j + 1], 1, &tau[j * l + i]);
                tau[j * l + i] = tau[j * l + i] / s[i];
                mvalue = -tau[j * l + i];
                cblas_zaxpy(n, &mvalue, r[i + 1], 1, r[j + 1], 1);
            }
            cblas_zdotc_sub(n, r[j + 1], 1, r[j + 1], 1, &s[j]);
            cblas_zdotc_sub(n, r[j + 1], 1, r[0], 1, &omega_d[j]);
            omega_d[j] = omega_d[j] / s[j];
        }

        omega[l - 1] = omega_d[l - 1];

        for (j = l - 2; j >= 0; j--) {
            t_local = 0;
            for (i = j + 1; i < l; i++) {
                t_local += tau[i * l + j] * omega[i];
            }
            omega[j] = omega_d[j] - t_local;
        }

        for (j = 0; j < l - 1; j++) {
            t_local = 0;
            for (i = j + 1; i < l - 1; i++) {
                t_local += tau[i * l + j] * omega[i + 1];
            }
            omega_dd[j] = omega[j + 1] + t_local;
        }

        // Update the seed system except the residual.
        cblas_zaxpy(n, &omega[0], r[0], 1, x, 1);
        mvalue = -omega[l - 1];
        cblas_zaxpy(n, &mvalue, p[l], 1, p[0], 1);
        for (i = 1; i < l; i++) {
            mvalue = -omega[i - 1];
            cblas_zaxpy(n, &mvalue, p[i], 1, p[0], 1);
            cblas_zaxpy(n, &omega_dd[i - 1], r[i], 1, x, 1);
        }

        // Shifted systems.
        for (k = 0; k < sigma_size; k++) {
            zhorner(omega, l, sigma[k], tmp);
            psi_shift = tmp[0];
            for (i = 0; i < l; i++) {
                omega_shift[i] = tmp[i + 1];
            }

            e_shift = theta_shift[k] * pi_shift[k];
            theta_shift[k] = theta_shift[k] * psi_shift;

            for (j = 0; j < l; j++) {
                tmp[0] = 0;
                for (i = j; i < l; i++) {
                    tmp[0] += M[k][i * l + j] * omega_shift[i];
                }
                omega_shift_d[j] = tmp[0];
            }

            for (j = 0; j < l - 1; j++) {
                tmp[0] = 0;
                for (i = j + 1; i < l - 1; i++) {
                    tmp[0] += tau[i * l + j] * omega_shift_d[i + 1];
                }
                omega_shift_dd[j] = omega_shift_d[j + 1] + tmp[0];
            }

            // Update shifted systems.
            mvalue = omega_shift_d[0] / e_shift;
            cblas_zaxpy(n, &mvalue, r[0], 1, x_shift[k], 1);
            mvalue = -omega[l - 1];
            cblas_zaxpy(n, &mvalue, p_shift[k][l], 1, p_shift[k][0], 1);
            for (i = 1; i < l; i++) {
                mvalue =  -omega[i - 1];
                cblas_zaxpy(n, &mvalue, p_shift[k][i], 1, p_shift[k][0], 1);
                mvalue = omega_shift_dd[i - 1] / e_shift;
                cblas_zaxpy(n, &mvalue, r[i], 1, x_shift[k], 1);
            }
            mvalue = 1.0 / psi_shift;
            cblas_zscal(n, &mvalue, p_shift[k][0], 1);
        }

        // Update the residual of the seed system.
        for (i = 1; i <= l; i++) {
            mvalue = -omega_d[i - 1];
            cblas_zaxpy(n, &mvalue, r[i], 1, r[0], 1);
        }

        error = cblas_dznrm2(n, r[0], 1) / norm_b;
        if (error < tol) {
            break;
        }
    }

    ////////// Finalization //////////
    free(tau);
    free(omega_base);
    free(base);
    free(p);
    free(r);
    free(small_mat_shift_base);
    free(omega_shift_base);
    free(theta_shift);
    free(p_shift);
    free(p_shift_ind);
    free(p_shift_base);
    free(x_shift);
    free(M_base);
    free(M);
    
    if (error >= tol) {
        return NOT_CONVERGED;
    }

    return iter;
}
