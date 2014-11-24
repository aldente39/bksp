/*****
    This program is solving a linear system
    with multiple right hand sides AX = B.
    In this program, the algorithm to solve it
    is the Block BiCGSTAB with orthogonalization of
    the direction matrix P.
*****/

#include <mkl.h>
#include <stdio.h>
#include "bksp.h"

int dblbicgstab (dspmat *A, dmat *B, double tol,
                        int max_iter, dmat *X) {
    int i, m, n;
    double omega, fnb, error;
    char *matdescra = "GLNC";
    double *alpha, *beta;
    double *P, *R, *Rs, *V, *T;
    int *ipiv;
    m = *A->row_size;
    n = B->col_size;
    error = 0;
    alpha = (double *)malloc(n * n * sizeof(double));
    beta = (double *)malloc(n * n * sizeof(double));
    P = (double *)malloc(m * n * sizeof(double));
    R = (double *)malloc(m * n * sizeof(double));
    Rs = (double *)malloc(m * n * sizeof(double));
    V = (double *)malloc(m * n * sizeof(double));
    T = (double *)malloc(m * n * sizeof(double));
    ipiv = (int *)malloc(n * n * sizeof(int));

    double *tmp = (double *) malloc(m * n * sizeof(double));
    double mone = -1;
    double zero = 0;
    dcsrmm0('n', m, n, A->value, A->I, A->J, *A->nnz, X->value, tmp);
    cblas_daxpy(m * n, 1, B->value, 1, tmp, 1);
    cblas_dcopy(m * n, tmp, 1, R, 1);
    cblas_dcopy(m * n, R, 1, P, 1);
    cblas_dcopy(m * n, R, 1, Rs, 1);
    fnb = cblas_dnrm2(m * n, B->value, 1);

    for (i = 0; i < max_iter; i++) {
        // Compute V = A * P.
        dcsrmm0('n', m, n, A->value, A->I, A->J, *A->nnz, P, V);

        // Compute alpha.
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    n, n, m, 1, Rs, m, V, m, 0, tmp, n);
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    n, n, m, 1, Rs, m, R, m, 0, alpha, n);
        LAPACKE_dgesv(LAPACK_COL_MAJOR, n, n, tmp, n, ipiv, alpha, n);

        // Compute the intermediate residual R.
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    m, n, n, -1, V, m, alpha, n, 1, R, m);

        // Compute T = A * R, where R is the intermediate residual.
        dcsrmm0('n', m, n, A->value, A->I, A->J, *A->nnz, R, T);

        // Compute omega.
        omega = cblas_ddot(m * n, T, 1, R, 1) /
                    cblas_ddot(m * n, T, 1, T, 1);

        // Compute the approximate solution X.
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    m, n, n, 1, P, m, alpha, n, 1, X->value, m);
        cblas_daxpy(m * n, omega, R, 1, X->value, 1);

        // Compute the residual R.
        cblas_daxpy(m * n, -omega, T, 1, R, 1);

        // Compute beta.
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    n, n, m, -1, Rs, m, T, m, 0, beta, n);
        LAPACKE_dgetrs(LAPACK_COL_MAJOR, 'n', n, n, tmp, n, ipiv, beta, n);

        // Compute the direction matrix P.
        cblas_daxpy(m * n, -omega, V, 1, P, 1);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    m, n, n, 1, P, m, beta, n, 0, tmp, m);
        cblas_daxpy(m * n, 1, R, 1, tmp, 1);
        cblas_dcopy(m * n, tmp, 1, P, 1);

        // Orthogonalize the direction matrix P.
        myqr(m, n, P, tmp);

        error = cblas_dnrm2(m * n, R, 1) / fnb;
        if (error < tol) {
            break;
        }
    }
    printf("%d, %e\n", i, error);

    free(tmp);
    free(ipiv);
    free(T);
    free(V);
    free(Rs);
    free(P);
    free(beta);
    free(alpha);
    return 0;
}

