/*****
    This is zero and one based csr format sparse matrix and
    column major dense matrix multiplication.
*****/

#include <stdio.h>
#include "bksp.h"

///// For zero-based indexing CSR matrix
int dcsrgemm0 (char type, int m, int n, double *A,
                int *Ai, int *Aj, int nnz, double *B, double *C) {
    int i, j, k;

    if (type == 'n') {
        #ifdef _OPENMP
        #pragma omp parallel for private(j, k)
        #endif
        for (i = 0; i < n; i++) {
            for (j = 0; j < m; j++) {
                C[i * m + j] = 0.0;
                for (k = Ai[j]; k < Ai[j + 1]; k++) {
                    C[i * m + j] = C[i * m + j] +
                                            A[k] * B[i * m + Aj[k]];
                }
            }
        }
    }
    else if (type == 't') {
        for (i = 0; i < n; i++) {
            for (j = 0; j < m; j++) {
                C[i * m + j] = 0;
            }
        }
        #ifdef _OPENMP
        #pragma omp parallel for private(j, k)
        #endif
        for (i = 0; i < n; i++) {
            for (j = 0; j < m; j++) {
                for (k = Ai[j]; k < Ai[j + 1]; k++) {
                    C[i * m + Aj[k]] = C[i * m + Aj[k]] +
                                            A[k] * B[i * m + j];
                }
            }
        }
    }
    else {
        printf("invalid parameter 0.\n");
        return -1;
    }

    return 0;
}

int dcsrsymm0 (char type, int m, int n, double *A,
             int *Ai, int *Aj, int nnz, double *B, double *C) {
    int i, j, k;

    if (type == 'l') {
        #ifdef _OPENMP
        #pragma omp parallel for private(j, k)
        #endif
        for (i = 0; i < n; i++) {
            for (j = 0; j < m; j++) {
                for (k = Ai[j]; k < Ai[j + 1]; k++) {
                    if (Aj[k] != j) {
                        C[i * m + Aj[k]] = C[i * m + Aj[k]] +
                            A[k] * B[i * m + j];
                    }
                    C[i * m + j] = C[i * m + j] +
                        A[k] * B[i * m + Aj[k]];
                }
            }
        }
    }
    else {
        printf("invalid parameter 0.\n");
        return -1;
    }

    return 0;
}

///// For one-based indexing CSR matrix
int dcsrgemm1 (char type, double alpha, int m, int n, double *A,
             int *Ai, int *Aj, int nnz, double *B, double beta, double *C) {
    int i, j, k;

    if (type == 'n') {
        #ifdef _OPENMP
        #pragma omp parallel for private(j, k)
        #endif
        for (i = 0; i < n; i++) {
            for (j = 0; j < m; j++) {
                C[i * m + j] = beta * C[i * m + j];
                for (k = Ai[j]-1; k < Ai[j + 1]-1; k++) {
                    C[i * m + j] = C[i * m + j] +
                        alpha * A[k] * B[i * m + Aj[k]-1];
                }
            }
        }
    }
    else if (type == 't') {
        for (i = 0; i < n; i++) {
            for (j = 0; j < m; j++) {
                C[i * m + j] = beta * C[i * m + j];
            }
        }
        #ifdef _OPENMP
        #pragma omp parallel for private(j, k)
        #endif
        for (i = 0; i < n; i++) {
            for (j = 0; j < m; j++) {
                for (k = Ai[j]-1; k < Ai[j + 1]-1; k++) {
                    C[i * m + Aj[k]-1] = C[i * m + Aj[k]-1] +
                        alpha * A[k] * B[i * m + j];
                }
            }
        }
    }
    else {
        printf("invalid parameter 0.\n");
        return -1;
    }

    return 0;
}

int dcsrsymm1 (char type, double alpha, int m, int n, double *A,
             int *Ai, int *Aj, int nnz, double *B, double beta, double *C) {
    int i, j, k;

    if (type == 'n') {
        #ifdef _OPENMP
        #pragma omp parallel for private(j, k)
        #endif
        for (i = 0; i < n; i++) {
            for (j = 0; j < m; j++) {
                for (k = Ai[j]-1; k < Ai[j + 1]-1; k++) {
                    if (Aj[k]-1 != j) {
                        C[i * m + Aj[k]-1] = C[i * m + Aj[k]-1] +
                            alpha * A[k] * B[i * m + j];
                    }
                    C[i * m + j] = C[i * m + j] +
                        alpha * A[k] * B[i * m + Aj[k]-1];
                }
            }
        }
    }
    else {
        printf("invalid parameter 0.\n");
        return -1;
    }

    return 0;
}

