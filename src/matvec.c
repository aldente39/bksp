/*****
    This is zero-based csr format sparse matrix and
    column major dense matrix multiplication.
*****/

#include <stdio.h>
#include "bksp.h"

int dcsrmm0 (char type, int m, int n, double *A,
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

