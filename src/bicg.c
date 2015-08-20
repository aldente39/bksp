#include <stdio.h>
#include <complex.h>
#include "bksp_internal.h"
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
                   mat->row, mat->col, x, tmp);
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
                     mat->row, mat->col, p, ap);

        // Compute A^{T} * sp.
        mkl_cspblas_dcsrgemv("t", mat->row_size, mat->value,
                     mat->row, mat->col, sp, atp);

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
    free(tmp);
    free(atp);
    free(ap);
    free(sp);
    free(p);
    free(sr);
    free(r);
    
    if (error >= tol) {
        return NOT_CONVERGED;
    }
    return i;
}


int zbicg (zspmat *mat, double _Complex *b,
            double tol, int max_iter, double _Complex *x) {
    ////////// Initialization //////////
    int i;
    double error, norm_b;
    double _Complex one = 1;
    double _Complex alpha, malpha, beta, mcalpha, cbeta, r_dot, r_dot_old;
    double _Complex *r, *sr, *p, *sp, *ap, *atp, *tmp;
    MKL_INT n = *mat->row_size;
    r = (double _Complex *) malloc(n * sizeof(double _Complex));
    sr = (double _Complex *) malloc(n * sizeof(double _Complex));
    p = (double _Complex *) malloc(n * sizeof(double _Complex));
    sp = (double _Complex *) malloc(n * sizeof(double _Complex));
    ap = (double _Complex *) malloc(n * sizeof(double _Complex));
    atp = (double _Complex *) malloc(n * sizeof(double _Complex));

    tmp = (double _Complex *) calloc(sizeof(double _Complex), n);
    mkl_cspblas_zcsrgemv("n", mat->row_size, mat->value,
                   mat->row, mat->col, x, tmp);
    cblas_zdscal(n, -1, tmp, 1);
    cblas_zaxpy(n, &one, b, 1, tmp, 1);
    cblas_zcopy(n, tmp, 1, r, 1);
    cblas_zcopy(n, tmp, 1, sr, 1);
    cblas_zcopy(n, r, 1, p, 1);
    cblas_zcopy(n, sr, 1, sp, 1);
    cblas_zdotc_sub(n, sr, 1, r, 1, &r_dot_old);
    norm_b = cblas_dznrm2(n, b, 1);

    ////////// Iteration //////////
    for (i = 0; i < max_iter; i++) {
        // Compute A * p.
        mkl_cspblas_zcsrgemv("n", mat->row_size, mat->value,
                     mat->row, mat->col, p, ap);

        // Compute A^{H} * sp.
        mkl_cspblas_zcsrgemv("t", mat->row_size, mat->value,
                     mat->row, mat->col, sp, atp);

        // Compute alpha.
        cblas_zdotc_sub(n, sp, 1, ap, 1, &alpha);
        alpha = r_dot_old / alpha;
        malpha = -(alpha);
        mcalpha = -conj(alpha);

        // Compute the approximate solution x.
        cblas_zaxpy(n, &alpha, p, 1, x, 1);

        // Compute the residual r.
        cblas_zaxpy(n, &malpha, ap, 1, r, 1);

        // Compute the shadow residual sr.
        cblas_zaxpy(n, &mcalpha, atp, 1, sr, 1);

        // Compute beta.
        cblas_zdotc_sub(n, sr, 1, r, 1, &r_dot);
        beta = r_dot / r_dot_old;
        cbeta = conj(beta);
        r_dot_old = r_dot;

        // Compute p.
        cblas_zscal(n, &beta, p, 1);
        cblas_zaxpy(n, &one, r, 1, p, 1);

        // Compute shadow p.
        cblas_zscal(n, &cbeta, sp, 1);
        cblas_zaxpy(n, &one, sr, 1, sp, 1);

        error = cblas_dznrm2(n, r, 1) / norm_b;
        if (error < tol) {
            break;
        }
    }

    ////////// Finalization //////////
    free(tmp);
    free(atp);
    free(ap);
    free(sp);
    free(p);
    free(sr);
    free(r);
    
    if (error >= tol) {
        return NOT_CONVERGED;
    }
    return i;
}
