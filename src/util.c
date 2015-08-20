#include <stdlib.h>
#include "bksp.h"
#include "bksp_internal.h"

dmat *dmat_create (int m, int n) {
    int i, l;
    l = m * n;
    dmat *mat = (dmat *)malloc(sizeof(dmat));
    double *value = (double *)malloc(m * n * sizeof(double));
    for (i = 0; i < l; i++) {
        value[i] = 0;
    }
    mat->row_size = m;
    mat->col_size = n;
    mat->value = value;

    return mat;
}

int dmat_free (dmat *mat) {
    free(mat->value);
    free(mat);

    return 0;
}

int darray2dmat (double *arr, int row_size, int col_size, dmat *mat) {
    mat->row_size = row_size;
    mat->col_size = col_size;
    mat->value = arr;

    return 0;
}

int copy_darray2mat (char type, double *arr, dmat *mat) {
    if (type == 'c') {
        int i, mn;
        mn = mat->row_size * mat->col_size;
        for (i = 0; i < mn; i++) {
            mat->value[i] = arr[i];
        }
    }
    else if (type == 'r') {
        int i, j, m, n;
        m = mat->row_size;
        n = mat->col_size;
        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++) {
                mat->value[(j * m) + i] = arr[(i * n) + j];
            }
        }
    }

    return 0;
}

int dvec2mat (double *x, int m, dmat *mat) {
    int i, l, n;
    l = mat->row_size;
    n = mat->col_size;

    for (i = 0; i < l; i++) {
        mat->value[i * n + m] = x[i];
    }

    return 0;
}


////////// For complex numbers //////////

zmat *zmat_create (int m, int n) {
    zmat *mat = (zmat *)malloc(sizeof(zmat));
    double _Complex *value = (double _Complex*)calloc(sizeof(double _Complex),
                                                      m * n);
    mat->row_size = m;
    mat->col_size = n;
    mat->value = value;

    return mat;
}

int zmat_free (zmat *mat) {
    free(mat->value);
    free(mat);

    return 0;
}
