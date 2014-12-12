#include <stdio.h>
#include <mkl.h>
#include "bksp.h"

// Horner's scheme
int *dhorner (double *gamma, int l, double sigma, double *out) {

    int i, j;
    out[l] = - gamma[l - 1];

    for (i = l - 1; i > 0; i--) {
        out[i] = - sigma * out[i + 1] - gamma[i];
    }

    out[0] = - sigma * out[1] + 1.0;

    for (i = 1; i < l; i++) {
        for (j = l - 1; j >= i; j--) {
            out[j] += - sigma * out[j + 1];
        }
    }

    for (i = 1; i <= l; i++) {
        out[i] = - out[i] / out[0];
    }

    return 0;
}

// Compute a combination
int comb (int n, int r) {
    if (n < 0 || r < 0 || n < r) {
        return 0;
    }
    if (n == r || !r) {
        return 1;
    }
    return comb(n - 1, r) + comb(n - 1, r - 1);
}

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
                         mat->I, mat->J, x, tmp);
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
                                 mat->I, mat->J, p[j], p[j + 1]);

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
                                 mat->I, mat->J, r[j], r[j + 1]);

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
                    tmp[0] += M[k][j * l + i] * omega_shift[i];
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

