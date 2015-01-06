/*****
    This program is solving a linear system
    with multiple right hand sides AX = B.
    The coefficient matrix A is symmetric.
    In this program, the algorithm to solve it
    is the Block CG method with orthogonalization of
    the residual matrix R.
*****/

#include "bksp_internal.h"
#include <stdio.h>
#include <string.h>
#include "bksp.h"

int dblcg (dspmat *A, dmat *B, double tol,
                        int max_iter, dmat *X) {

    if (strcmp(A->type, "symmetric") != 0) {
        return INVALID_MATRIX_TYPE;
    }

    ////////// Initialization //////////
    int i, m, n;
    double fnb, error;
    char *matdescra = "SLNF";
    double *alpha, *beta, *C, *base0;
    double *P, *V, *ap, *tmp, *base1;
    int *row_ptr, *col_ind, *ipiv;
    m = *A->row_size;
    n = B->col_size;
    error = 0;
    base0 = (double *)malloc(n * n * 3 * sizeof(double));
    alpha = base0;
    beta = &base0[n * n];
    C = &base0[n * n * 2];
    base1 = (double *)malloc(m * n * 4 * sizeof(double));
    P = base1;
    V = &base1[m * n];
    ap = &base1[m * n * 2];
    tmp = &base1[m * n * 3];
    ipiv = (int *)malloc(n * n * sizeof(double));
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
    mkl_dcsrmm("n", &m, &n, &m, &mone, matdescra, A->value, col_ind,
               row_ptr, &row_ptr[1], X->value, &m, &zero, tmp, &m);
    cblas_daxpy(m * n, 1, B->value, 1, tmp, 1);
    cblas_dcopy(m * n, tmp, 1, V, 1);
    myqr(m, n, V, C);
    cblas_dcopy(m * n, V, 1, P, 1);
    fnb = cblas_dnrm2(m * n, B->value, 1);
    for (i = 0; i < n; i++) {
        beta[i * n + i] = 1.0;
    }

    ////////// Iteration //////////
    for (i = 0; i < max_iter; i++) {
        // Compute the direction matrix P.
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    m, n, n, 1, P, m, beta, n, 0, tmp, m);
        cblas_daxpy(m * n, 1, V, 1, tmp, 1);
        cblas_dcopy(m * n, tmp, 1, P, 1);

        // Compute A * P.
        mkl_dcsrmm("n", &m, &n, &m, &one, matdescra, A->value, col_ind,
                   row_ptr, &row_ptr[1], P, &m, &zero, ap, &m);

        // Compute alpha.
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    n, n, m, 1, P, m, ap, m, 0, alpha, n);
        // Compute inverse.
        LAPACKE_dgetrf(LAPACK_COL_MAJOR, n, n, alpha, n, ipiv);
        LAPACKE_dgetri(LAPACK_COL_MAJOR, n, alpha, n, ipiv);

        // Compute the approximate solution X.
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n, n, n, 1, alpha, n, C, n, 0, tmp, n);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    m, n, n, 1, P, m, tmp, n, 1, X->value, m);

        // Compute the residual matrix V.
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    m, n, n, -1, ap, m, alpha, n, 1, V, m); 

        // Orthogonalize the residual matrix V.
        myqr(m, n, V, beta);

        // Compute C.
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n, n, n, 1, beta, n, C, n, 0, tmp, n);
        cblas_dcopy(n * n, tmp, 1, C, 1);

        error = cblas_dnrm2(n * n, C, 1) / fnb;
        if (error < tol) {
            break;
        }
    }

    ////////// Finalization //////////
    free(base1);
    free(base0);
    free(ipiv);
    free(col_ind);
    free(row_ptr);
    
    if (error >= tol) {
        return NOT_CONVERGED;
    }
    return i;
}

