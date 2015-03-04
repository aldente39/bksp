/*****
    This program is solving a linear system
    with multiple right hand sides AX = B.
    In this program, the algorithm to solve it
    is the Block BiCGSTAB with orthogonalization of
    the direction matrix P.
*****/

#include "bksp_internal.h"
#include <stdio.h>
#include "bksp.h"

int dblbicgstab (dspmat *A, dmat *B, double tol,
                        int max_iter, dmat *X) {
    ////////// Initialization //////////
    int i, m, n;
    double omega, fnb, error;
    char *matdescra = "GLNF";
    double *alpha, *beta, *base0;
    double *P, *R, *Rs, *V, *T, *tmp, *base1;
    int *row_ptr, *col_ind, *ipiv;
    m = *A->row_size;
    n = B->col_size;
    error = 0;
    base0 = (double *)malloc(n * n * 2 * sizeof(double));
    alpha = base0;
    beta = &base0[n * n];
    base1 = (double *)malloc(m * n * 6 * sizeof(double));
    P = base1;
    R = &base1[m * n];
    Rs = &base1[m * n * 2];
    V = &base1[m * n * 3];
    T = &base1[m * n * 4];
    tmp = &base1[m * n * 5];
    ipiv = (int *)malloc(n * n * sizeof(int));
    row_ptr = (int *)malloc((m + 1) * sizeof(int));
    col_ind = (int *)malloc((*A->nnz) * sizeof(int));
    for (i = 0; i <= m; i++) {
        row_ptr[i] = A->row[i] + 1;
    }
    for (i = 0; i < *A->nnz; i++) {
        col_ind[i] = A->col[i] + 1;
    }

    double one = 1;
    double mone = -1;
    double zero = 0;
    //dcsrmm0('n', m, n, A->value, A->row, A->col, *A->nnz, X->value, tmp);
    mkl_dcsrmm("n", &m, &n, &m, &mone, matdescra, A->value,
               col_ind, row_ptr, &row_ptr[1], X->value, &m, &zero, tmp, &m);
    cblas_daxpy(m * n, 1, B->value, 1, tmp, 1);
    cblas_dcopy(m * n, tmp, 1, R, 1);
    cblas_dcopy(m * n, R, 1, P, 1);
    cblas_dcopy(m * n, R, 1, Rs, 1);
    fnb = cblas_dnrm2(m * n, B->value, 1);

    ////////// Iteration //////////
    for (i = 0; i < max_iter; i++) {
        // Compute V = A * P.
        //dcsrmm0('n', m, n, A->value, A->row, A->col, *A->nnz, P, V);
        mkl_dcsrmm("n", &m, &n, &m, &one, matdescra, A->value,
            col_ind, row_ptr, &row_ptr[1], P, &m, &zero, V, &m);

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
        mkl_dcsrmm("n", &m, &n, &m, &one, matdescra, A->value,
            col_ind, row_ptr, &row_ptr[1], R, &m, &zero, T, &m);

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

    ////////// Finalization //////////
    free(row_ptr);
    free(col_ind);
    free(ipiv);
    free(base1);
    free(base0);
    
    if (error >= tol) {
        return NOT_CONVERGED;
    }
    return i;
}

