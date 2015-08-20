/*****
    This program is solving a linear system
    with multiple right hand sides AX = B.
    In this program, the algorithm to solve it
    is the Block BiCGSTAB(l) with orthogonalization of
    the residual matrix R and multiplying the coefficient
    matrix A explicitly (I call it "Block BiCGSTAB(l)-RH").
*****/

#include "bksp_internal.h"
#include <stdio.h>
#include <math.h>
#include "bksp.h"


int dblbicgstabl (dspmat *A, dmat *B, int l, double tol,
                        int max_iter, dmat *X) {
    // ********** Initialization **********
    int iter, i, j, m, n;
    double fnb, error, t_local;
    char *matdescra = "GLNF";
    double *alpha, *beta, *t, *xi;
    double *gamma0, *gamma1, *omega, *omega_d, *omega_dd, *sigma, *tau;
    double **P, **R, *Rs, **V, **PV;
    double *R_base, *PV_base;
    int *row_ptr, *col_ind;
    int *ipiv;
    m = dmat_rowsize(B);
    n = dmat_colsize(B);
    error = 0;
    alpha = (double *)calloc(n * n, sizeof(double));
    beta = (double *)malloc(n * n * sizeof(double));
    t = (double *)malloc(n * n * sizeof(double));
    xi = (double *)calloc(n * n, sizeof(double));
    gamma0 = (double *)malloc(n * n * sizeof(double));
    gamma1 = (double *)malloc(n * n * sizeof(double));
    omega = (double *)malloc((l + 1) * sizeof(double));
    omega[l] = 1;
    omega_d = (double *)malloc((l + 1) * sizeof(double));
    omega_dd = (double *)malloc((l + 1) * sizeof(double));
    sigma = (double *)malloc((l + 1) * sizeof(double));
    tau = (double *) malloc((l + 1) * (l + 1) * sizeof(double));
    P = (double **)malloc((l + 1) * sizeof(double));
    R = (double **)malloc((l + 1) * sizeof(double));
    PV = (double **)malloc(2 * (l + 1) * sizeof(double));
    Rs = (double *)malloc(m * n * sizeof(double));
    V = (double **)malloc((l + 1) * sizeof(double));
    R_base = (double *)malloc((l + 1) * m * n * sizeof(double));
    //PV_base = (double *)malloc(2 * (l + 1) * m * n * sizeof(double));
    PV_base = (double *)calloc(2 * (l + 1) * m * n, sizeof(double));
    row_ptr = (int *)malloc((m + 1) * sizeof(int));
    col_ind = (int *)malloc((*A->nnz) * sizeof(int));
    for (i = 0; i < m + 1; i++) {
        row_ptr[i] = A->row[i] + 1;
    }
    for (i = 0; i < *A->nnz; i++) {
        col_ind[i] = A->col[i] + 1;
    }
    for (i = 0; i < 2 * (l + 1); i++) {
        PV[i] = &PV_base[i * m * n];
    }
    for (i = 0; i <= l; i++) {
        P[i] = PV[2 * i];
        V[i] = PV[2 * i + 1];
        R[i] = &R_base[i * m * n];
    }
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (j == i) {
                gamma0[i * n + j] = 1;
            }
            else {
                gamma0[i * n + j] = 0;
            }
        }
    }
    ipiv = (int *)malloc(n * n * sizeof(int));

    double *tmp = (double *)calloc(m * n, sizeof(double));
    double one = 1;
    double mone = -1;
    double zero = 0;
    mkl_dcsrmm("n", &m, &n, &m, &mone, matdescra, A->value,
            col_ind, row_ptr, &row_ptr[1], X->value, &m, &zero, tmp, &m);
    cblas_dscal(m * n, -1, tmp, 1);
    cblas_daxpy(m * n, 1, B->value, 1, tmp, 1);
    cblas_dcopy(m * n, tmp, 1, V[0], 1);
    int info = myqr(m, n, V[0], xi);
    cblas_dcopy(m * n, V[0], 1, P[0], 1);
    cblas_dcopy(m * n, V[0], 1, Rs, 1);

    // Compute the frobenius norm of B.
    fnb = cblas_dnrm2(m * n, B->value, 1);

    // ********** Outer teration **********
    for (iter = 0; iter < max_iter; iter++) {
        // ********** BiCG Part **********
        cblas_dscal(n * n, -omega[l], gamma0, 1);

        for (j = 0; j < l; j++) {
            // Compute beta.
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                            n, n, m, 1, Rs, m, V[j], m, 0, gamma1, n);
            cblas_dcopy(n * n, gamma1, 1, tmp, 1);
            LAPACKE_dgesv(LAPACK_COL_MAJOR, n, n, gamma0, n, ipiv, tmp, n);
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            n, n, n, -1, alpha, n, tmp, n, 0, beta, n);
            cblas_dcopy(n * n, gamma1, 1, gamma0, 1);

            // Compute the direction matrix P[i].
            for (i = 0; i <= j; i++) {
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            m, n, n, 1, P[i], m, beta, n, 0, tmp, m);
                cblas_daxpy(m * n, 1, V[i], 1, tmp, 1);
                cblas_dcopy(m * n, tmp, 1, P[i], 1);
            }
            mkl_dcsrmm("n", &m, &n, &m, &one, matdescra, A->value, col_ind,
                    row_ptr, &row_ptr[1], P[j], &m, &zero, P[j + 1], &m);

            // Compute alpha.
            cblas_dcopy(n * n, gamma1, 1, alpha, 1);
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                            n, n, m, 1, Rs, m, P[j + 1], m, 0, tmp, n);
            LAPACKE_dgesv(LAPACK_COL_MAJOR, n, n, tmp, n, ipiv, alpha, n);

            // Compute the residual matrix V[0].
            for (i = 0; i <= j; i++) {
                if (i == 0) {
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            m, n, n, 1, P[0], m, alpha, n, 0, tmp, m);
                    mkl_dcsrmm("n", &m, &n, &m, &mone, matdescra, A->value,
                    col_ind, row_ptr, &row_ptr[1], tmp, &m, &one, V[0], &m);

                    // Orthogonalize the residual matrix V[0].
                    myqr(m, n, V[0], t);

                    // Compute the approximate solution X.
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                m, n, n, 1, tmp, m, xi, n, 1, X->value, m);

                    // Compute the upper triangler matrix xi.
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                n, n, n, 1, t, n, xi, n, 0, tmp, n);
                    cblas_dcopy(n * n, tmp, 1, xi, 1);

                    // Compute inverse.
                    LAPACKE_dgetrf(LAPACK_COL_MAJOR, n, n, t, n, ipiv);
                    LAPACKE_dgetri(LAPACK_COL_MAJOR, n, t, n, ipiv);
                }
                else {
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            m, n, n, -1, P[i + 1], m, alpha, n, 1, V[i], m);
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            m, n, n, 1, V[i], m, t, n, 0, tmp, m);
                    cblas_dcopy(m * n, tmp, 1, V[i], 1);
                }
            }
            
            mkl_dcsrmm("n", &m, &n, &m, &one, matdescra, A->value, col_ind,
                    row_ptr, &row_ptr[1], V[j], &m, &zero, V[j + 1], &m);
        }

        for (i = 0; i <= l; i++) {
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        m, n, n, 1, V[i], m, xi, n, 0, R[i], m);
        }


        // ********** MR Part **********
        for (j = 1; j <= l; j++) {
            for (i = 1; i < j; i++) {
                tau[i * l + j] = cblas_ddot(m * n, R[j], 1, R[i], 1) /
                                        sigma[i];
                cblas_daxpy(m * n, -tau[i * l + j], R[i], 1, R[j], 1);
            }
            sigma[j] = cblas_ddot(m * n, R[j], 1, R[j], 1);
            omega_d[j] = cblas_ddot(m * n, R[0], 1, R[j], 1) / sigma[j];
        }
        omega[l] = omega_d[l];

        for (j = l - 1; j > 0; j--) {
            t_local = 0;
            for (i = j + 1; i <= l; i++) {
                t_local += tau[j * l + i] * omega[i];
            }
            omega[j] = omega_d[j] - t_local;
        }

        /*for (j = 1; j < l; j++) {
            t_local = 0;
            for (i = j + 1; i < l; i++) {
                t_local += tau[j * l + i] * omega[i + 1];
            }
            omega_dd[j] = omega[j + 1] + t_local;
        }*/

        // ********** Update **********
        cblas_dscal(m * n, 0, tmp, 1);
        for (i = 1; i <= l; i++) {
            cblas_daxpy(m * n, omega[i], V[i - 1], 1, tmp, 1);
            cblas_daxpy(m * n, -omega[i], P[i], 1, P[0], 1);
            //cblas_daxpy(m * n, -omega[i], V[i], 1, V[0], 1);
        }
        mkl_dcsrmm("n", &m, &n, &m, &mone, matdescra, A->value, col_ind,
                    row_ptr, &row_ptr[1], tmp, &m, &one, V[0], &m);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            m, n, n, 1, tmp, m, xi, n, 1, X->value, m);

        error = cblas_dnrm2(n * n, xi, 1) / fnb;
        if (error < tol || isnan(error)) {
            break;
        }
    }

    // Compute Frobenius norm of the true residual.
    mkl_dcsrmm("n", &m, &n, &m, &mone, matdescra, A->value,
            col_ind, row_ptr, &row_ptr[1], X->value, &m, &zero, tmp, &m);
    cblas_daxpy(m * n, 1, B->value, 1, tmp, 1);

    printf("Block BiCGSTAB(%d) has converged at:\n\
        The number of iteration ... %d\n\
        Frobenius norm of the relative residual ... %e\n\
        Frobenius norm of the true residual ... %e\n",
           l, iter, error, cblas_dnrm2(m * n, tmp, 1) / fnb);

    // ********** Finalization **********
    free(tmp);
    free(ipiv);
    free(tau);
    free(Rs);
    free(P);
    free(R);
    free(V);
    free(PV);
    free(beta);
    free(alpha);
    free(sigma);
    free(gamma1);
    free(gamma0);
    free(t);
    free(xi);
    free(omega_dd);
    free(omega_d);
    free(omega);
    free(R_base);
    free(PV_base);
    
    if (error >= tol) {
        return NOT_CONVERGED;
    }
    return iter;
}


int zblbicgstabl (zspmat *A, zmat *B, int l, double tol,
                        int max_iter, zmat *X) {
    // ********** Initialization **********
    int iter, i, j, m, n;
    double fnb, error, t_local;
    char *matdescra = "GLNF";
    double _Complex *alpha, *beta, *t, *xi, *small_mat_base;
    double _Complex *gamma0, *gamma1;
    double _Complex *omega_base, *omega, *omega_d, *omega_dd, *sigma, *tau;
    double _Complex **P, **R, *Rs, **V, **PV, **index_base;
    double _Complex *R_base, *PV_base, *tmp;
    int *row_ptr, *col_ind;
    int *ipiv;
    double _Complex one, mone, zero, mvalue;
    m = zmat_rowsize(B);
    n = zmat_colsize(B);
    error = 0;
    small_mat_base = (double _Complex *)calloc(n * n * 6,
                                               sizeof(double _Complex));
    alpha = small_mat_base;
    beta = &small_mat_base[n * n];
    t = &small_mat_base[n * n * 2];
    xi = &small_mat_base[n * n * 3];
    gamma0 = &small_mat_base[n * n * 4];
    gamma1 = &small_mat_base[n * n * 5];
    //alpha = (double *)calloc(sizeof(double _Complex), n * n);
    //beta = (double *)malloc(n * n * sizeof(double _Complex));
    //t = (double *)malloc(n * n * sizeof(double _Complex));
    //xi = (double *)malloc(n * n * sizeof(double _Complex));
    //gamma0 = (double *)malloc(n * n * sizeof(double _Complex));
    //gamma1 = (double *)malloc(n * n * sizeof(double _Complex));
    omega_base = (double _Complex *)malloc(3 * l * sizeof(double _Complex));
    omega = omega_base;
    omega[l - 1] = 1;
    omega_d = &omega_base[l];
    omega_dd = &omega_base[l * 2];
    sigma = (double _Complex *)malloc(l * sizeof(double _Complex));
    tau = (double _Complex *)calloc(l * l, sizeof(double _Complex));
    index_base = (double _Complex **)malloc(sizeof(double _Complex *) * (l + 1) * 5);
    P = index_base;
    R = &index_base[l + 1];
    V = &index_base[(l + 1) * 2];
    PV = &index_base[(l + 1) * 3];
    //P = (double **)malloc((l + 1) * sizeof(double));
    //R = (double **)malloc((l + 1) * sizeof(double));
    //PV = (double **)malloc(2 * (l + 1) * sizeof(double));
    Rs = (double _Complex *)malloc(m * n * sizeof(double _Complex));
    //V = (double **)malloc((l + 1) * sizeof(double));
    R_base = (double _Complex *)calloc((l + 1) * m * n, sizeof(double _Complex));
    PV_base = (double _Complex *)calloc(2 * (l + 1) * m * n,
                                        sizeof(double _Complex));
    row_ptr = (int *)malloc((m + 1) * sizeof(int));
    col_ind = (int *)malloc((*A->nnz) * sizeof(int));
    for (i = 0; i < m + 1; i++) {
        row_ptr[i] = A->row[i] + 1;
    }
    for (i = 0; i < *A->nnz; i++) {
        col_ind[i] = A->col[i] + 1;
    }
    for (i = 0; i < 2 * (l + 1); i++) {
        PV[i] = &PV_base[i * m * n];
    }
    for (i = 0; i <= l; i++) {
        P[i] = PV[2 * i];
        V[i] = PV[2 * i + 1];
        R[i] = &R_base[i * m * n];
    }
    for (i = 0; i < n; i++) {
        gamma0[i * n + i] = 1.0;
    }
    ipiv = (int *)malloc(n * n * sizeof(int));

    tmp = (double _Complex *)calloc(sizeof(double _Complex), m * n);
    one = 1;
    mone = -1;
    zero = 0;
    mkl_zcsrmm("n", &m, &n, &m, &mone, matdescra, A->value,
            col_ind, row_ptr, &row_ptr[1], X->value, &m, &zero, tmp, &m);
    cblas_zdscal(m * n, -1, tmp, 1);
    cblas_zaxpy(m * n, &one, B->value, 1, tmp, 1);
    cblas_zcopy(m * n, tmp, 1, V[0], 1);
    int info = zqreco(m, n, V[0], xi);
    cblas_zcopy(m * n, V[0], 1, P[0], 1);
    cblas_zcopy(m * n, V[0], 1, Rs, 1);
    //int rand_seed[4] = {1,0,3,2};
    //LAPACKE_zlarnv(1, rand_seed, m * n, Rs);

    // Compute the frobenius norm of B.
    fnb = cblas_dznrm2(m * n, B->value, 1);

    // ********** Outer teration **********
    for (iter = 0; iter < max_iter; iter++) {
        // ********** BiCG Part **********
        mvalue = -omega[l - 1];
        cblas_zscal(n * n, &mvalue, gamma0, 1);

        for (j = 0; j < l; j++) {
            // Compute beta.
            cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                            n, n, m, &one, Rs, m, V[j], m, &zero, gamma1, n);
            cblas_zcopy(n * n, gamma1, 1, tmp, 1);
            LAPACKE_zgesv(LAPACK_COL_MAJOR, n, n, gamma0, n, ipiv, tmp, n);
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            n, n, n, &one, alpha, n, tmp, n, &zero, beta, n);
            cblas_zcopy(n * n, gamma1, 1, gamma0, 1);

            // Compute the direction matrix P[i].
            for (i = 0; i <= j; i++) {
                cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            m, n, n, &mone, P[i], m, beta, n, &zero, tmp, m);
                cblas_zaxpy(m * n, &one, V[i], 1, tmp, 1);
                cblas_zcopy(m * n, tmp, 1, P[i], 1);
            }
            mkl_zcsrmm("n", &m, &n, &m, &one, matdescra, A->value, col_ind,
                    row_ptr, &row_ptr[1], P[j], &m, &zero, P[j + 1], &m);

            // Compute alpha.
            cblas_zcopy(n * n, gamma1, 1, alpha, 1);
            cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                            n, n, m, &one, Rs, m, P[j + 1], m, &zero, tmp, n);
            LAPACKE_zgesv(LAPACK_COL_MAJOR, n, n, tmp, n, ipiv, alpha, n);

            // Compute the residual matrix V[0].
            for (i = 0; i <= j; i++) {
                if (i == 0) {
                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            m, n, n, &one, P[0], m, alpha, n, &zero, tmp, m);
                    mkl_zcsrmm("n", &m, &n, &m, &mone, matdescra, A->value,
                    col_ind, row_ptr, &row_ptr[1], tmp, &m, &one, V[0], &m);

                    // Orthogonalize the residual matrix V[0].
                    info = zqreco(m, n, V[0], t);

                    // Compute the approximate solution X.
                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                m, n, n, &one, tmp, m,
                                xi, n, &one, X->value, m);

                    // Compute the upper triangler matrix xi.
                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                n, n, n, &one, t, n, xi, n, &zero, tmp, n);
                    cblas_zcopy(n * n, tmp, 1, xi, 1);

                    // Compute inverse.
                    LAPACKE_zgetrf(LAPACK_COL_MAJOR, n, n, t, n, ipiv);
                    LAPACKE_zgetri(LAPACK_COL_MAJOR, n, t, n, ipiv);
                }
                else {
                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        m, n, n, &mone, P[i + 1], m, alpha, n, &one, V[i], m);
                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            m, n, n, &one, V[i], m, t, n, &zero, tmp, m);
                    cblas_zcopy(m * n, tmp, 1, V[i], 1);
                }
            }
            
            mkl_zcsrmm("n", &m, &n, &m, &one, matdescra, A->value, col_ind,
                    row_ptr, &row_ptr[1], V[j], &m, &zero, V[j + 1], &m);
        }

        for (i = 0; i <= l; i++) {
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        m, n, n, &one, V[i], m, xi, n, &zero, R[i], m);
        }


        // ********** MR Part **********
        for (j = 0; j < l; j++) {
            for (i = 0; i < j; i++) {
                cblas_zdotc_sub(m * n, R[i+1], 1, R[j+1], 1, &tau[j * l + i]);
                tau[j * l + i] = tau[j * l + i] / sigma[i];
                mvalue = -tau[j * l + i];
                cblas_zaxpy(m * n, &mvalue, R[i + 1], 1, R[j + 1], 1);
            }
            cblas_zdotc_sub(m * n, R[j+1], 1, R[j+1], 1, &sigma[j]);
            cblas_zdotc_sub(m * n, R[j+1], 1, R[0], 1, &omega_d[j]);
            omega_d[j] = omega_d[j] / sigma[j];
        }
        omega[l - 1] = omega_d[l - 1];

        for (j = l - 2; j >= 0; j--) {
            t_local = 0;
            for (i = j + 1; i < l; i++) {
                t_local += tau[i * l + j] * omega[i];
            }
            omega[j] = omega_d[j] - t_local;
        }

        /*for (j = 1; j < l; j++) {
            t_local = 0;
            for (i = j + 1; i < l; i++) {
                t_local += tau[j * l + i] * omega[i + 1];
            }
            omega_dd[j] = omega[j + 1] + t_local;
        }*/

        // ********** Update **********
        cblas_zdscal(m * n, 0, tmp, 1);
        for (i = 0; i < l; i++) {
            mvalue = -omega[i];
            cblas_zaxpy(m * n, &omega[i], V[i], 1, tmp, 1);
            cblas_zaxpy(m * n, &mvalue, P[i + 1], 1, P[0], 1);
            //cblas_daxpy(m * n, -omega[i], V[i], 1, V[0], 1);
        }
        mkl_zcsrmm("n", &m, &n, &m, &mone, matdescra, A->value, col_ind,
                    row_ptr, &row_ptr[1], tmp, &m, &one, V[0], &m);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            m, n, n, &one, tmp, m, xi, n, &one, X->value, m);

        error = cblas_dznrm2(n * n, xi, 1) / fnb;
        if (error < tol || isnan(error)) {
            break;
        }
    }

    // Compute Frobenius norm of the true residual.
    mkl_zcsrmm("n", &m, &n, &m, &mone, matdescra, A->value,
            col_ind, row_ptr, &row_ptr[1], X->value, &m, &zero, tmp, &m);
    cblas_zaxpy(m * n, &one, B->value, 1, tmp, 1);

    printf("Block BiCGSTAB(%d) has converged at:\n\
        The number of iteration ... %d\n\
        Frobenius norm of the relative residual ... %e\n\
        Frobenius norm of the true residual ... %e\n",
           l, iter, error, cblas_dznrm2(m * n, tmp, 1) / fnb);

    // ********** Finalization **********
    free(tmp);
    free(ipiv);
    free(tau);
    free(Rs);
    //free(P);
    //free(R);
    //free(V);
    //free(PV);
    free(index_base);
    //free(beta);
    //free(alpha);
    //free(sigma);
    //free(gamma1);
    //free(gamma0);
    //free(t);
    //free(xi);
    free(small_mat_base);
    free(omega_base);
    free(R_base);
    free(PV_base);
    
    if (error >= tol || isnan(error)) {
        return NOT_CONVERGED;
    }
    return iter;
}
