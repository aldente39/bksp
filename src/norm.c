#include <string.h>
#include <math.h>
#include "bksp.h"

double sp_norm_f_unsymmetric (dspmat *A) {
    double n = 0;
    int i;
    int end = *A->nnz;
    for (i = 0; i < end; i++) {
        n += A->value[i] * A->value[i];
    }

    return sqrt(n);
}

double sp_norm_f_symmetric(dspmat *A) {
    double n = 0;
    int i, j;
    double tmp;
    int *row_ind = A->row;
    int *col_ptr = A->col;
    int size = *A->row_size;
    for(i = 0; i < size; i++) {
        for (j = row_ind[i]; j < row_ind[i + 1]; j++) {
            tmp = A->value[j] * A->value[j];
            if (col_ptr[j] == i) {
                n += tmp;
            }
            else {
                n += 2 * tmp;
            }
        }
    }
    return sqrt(n);
}

double sp_norm_f(dspmat *A) {
    if (strcmp(A->type, "symmetric") == 0) {
        return sp_norm_f_symmetric(A);
    }
    else if (strcmp(A->type, "unsymmetric") == 0) {
        return sp_norm_f_unsymmetric(A);
    }
    else {
        return -1.0;
    }
}

