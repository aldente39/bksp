#include <stdio.h>
#include <mkl.h>
#include "bksp.h"

int dshbicg (dspmat *mat, double *b, double *sigma,
            int sigma_size, double tol, int max_iter, double *x) {
    // ********** Initialization **********
    int i, k;
    double error = 0;
    double alpha, beta, alpha_old, rho, rho_old, fnb;
    double *r, *sr, *p, *sp, *ap, *atp, *tmp, *base;
    MKL_INT n = *mat->row_size;
    base = (double *) calloc(n * 7, sizeof(double));
    r = base;
    sr = &base[n];
    p = &base[n * 2];
    sp = &base[n * 3];
    ap = &base[n * 4];
    atp = &base[n * 5];
    tmp = &base[n * 6];
    mkl_cspblas_dcsrgemv("n", mat->row_size, mat->value,
                   mat->I, mat->J, x, tmp);
    cblas_dscal(n, -1, tmp, 1);
    cblas_daxpy(n, 1, b, 1, tmp, 1);
    cblas_dcopy(n, tmp, 1, r, 1);
    cblas_dcopy(n, tmp, 1, sr, 1);
    fnb = cblas_dnrm2(n, b, 1);
    alpha_old = 1;
    rho_old = 1;
    
    // For shifted systems
    double *shift_base = (double *)malloc(sigma_size * 5 * sizeof(double));
    double *alpha_shift = shift_base;
    double *beta_shift = &shift_base[sigma_size];
    double *pi_shift = &shift_base[sigma_size * 2];
    double *pi_shift_new = &shift_base[sigma_size * 3];
    double *pi_shift_old = &shift_base[sigma_size * 4];
    double **p_shift = (double **)malloc(sigma_size * sizeof(double *));
    double **x_shift = (double **)malloc(sigma_size * sizeof(double *));
    double *p_shift_base = (double *)calloc(n * sigma_size, sizeof(double));
    for (i = 0; i < sigma_size; i++) {
        p_shift[i] = &p_shift_base[i * n];
        x_shift[i] = &x[(i + 1) * n];
        pi_shift_old[i] = 1;
        pi_shift[i] = 1;
    }

    // ********** Iteration **********
    for (i = 0; i < max_iter; i++) {
        
        // Compute beta.
        rho = cblas_ddot(n, r, 1, sr, 1);
        beta = - rho / rho_old;
        
        // Compute p.
        cblas_dscal(n, - beta, p, 1);
        cblas_daxpy(n, 1, r, 1, p, 1);
        
        // Compute p^{*}.
        cblas_dscal(n, - beta, sp, 1);
        cblas_daxpy(n, 1, sr, 1, sp, 1);
        
        // Compute A * p.
        mkl_cspblas_dcsrgemv("n", mat->row_size, mat->value,
                     mat->I, mat->J, p, ap);
        
        // Compute A^{T} * p^{*}.
        mkl_cspblas_dcsrgemv("t", mat->row_size, mat->value,
                     mat->I, mat->J, sp, atp);
        
        // Compute alpha.
        alpha = rho / cblas_ddot(n, ap, 1, sr, 1);
        
        // Compute x.
        cblas_daxpy(n, alpha, p, 1, x, 1);
        
        // Shifted systems.
        for (k = 0; k < sigma_size; k++) {
            // Compute pi_shift_new.
            pi_shift_new[k] = (1 + alpha * sigma[k]) * pi_shift[k]
                + alpha * beta / alpha_old * (pi_shift_old[k] - pi_shift[k]);
            
            // Compute beta_shift.
            beta_shift[k] = (pi_shift_old[k] / pi_shift[k]) *
                (pi_shift_old[k] / pi_shift[k]) * beta;
            
            // Compute alpha_shift.
            alpha_shift[k] = pi_shift[k] / pi_shift_new[k] * alpha;

            // Compute p_shift.
            cblas_dscal(n, - beta_shift[k], p_shift[k], 1);
            cblas_daxpy(n, 1.0 / pi_shift[k], r, 1, p_shift[k], 1);
            
            // Compute x_shift.
            cblas_daxpy(n, alpha_shift[k], p_shift[k], 1, x_shift[k], 1);
        }
        
        // Compute r.
        cblas_daxpy(n, -alpha, ap, 1, r, 1);
        
        // Compute r^{*}.
        cblas_daxpy(n, -alpha, atp, 1, sr, 1);
        
        // Update scalars.
        alpha_old = alpha;
        rho_old = rho;
        for (k = 0; k < sigma_size; k++) {
            pi_shift_old[k] = pi_shift[k];
            pi_shift[k] = pi_shift_new[k];
        }

        error = cblas_dnrm2(n, r, 1) / fnb;
        if (error < tol) {
            break;
        }
    }

    // ********** Finalization **********
    free(base);
    free(shift_base);
    free(p_shift);
    free(x_shift);
    free(p_shift_base);
    
    if (error >= tol) {
        return NOT_CONVERGED;
    }
    return i;
}

