#include "bksp_internal.h"
//#include "bksp.h"

int myqr (int m, int n, double *mat, double *r) {
    int i, j, ind, info;
    int half = 0;
    double *tau = malloc(n * n * sizeof(double));

    info = LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, mat, m, tau);

    for (i = 0; i < n; i++) {
        half = (n + 1) * i;
        for (j = 0; j < n; j++) {
            ind = (i * n) + j;
            if (ind > half) {
                r[ind] = 0;
            }
            else {
                r[ind] = mat[i * m + j];
            }
        }
    }
    /*for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%e ",r[j * n + i]);
        }
        printf("\n");
    }*/

    info = LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, n, n, mat, m, tau);

    free(tau);
    if (info != 0) {
        return info;
    }
    else {
        return 0;
    }
}

int zqreco (int m, int n, double _Complex *mat, double _Complex *r) {
    int i, j, ind, info;
    int half = 0;
    double _Complex *tau = malloc(n * n * sizeof(double _Complex));

    info = LAPACKE_zgeqrf(LAPACK_COL_MAJOR, m, n, mat, m, tau);

    for (i = 0; i < n; i++) {
        half = (n + 1) * i;
        for (j = 0; j < n; j++) {
            ind = (i * n) + j;
            if (ind > half) {
                r[ind] = 0;
            }
            else {
                r[ind] = mat[i * m + j];
            }
        }
    }
    /*for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%f+%fi ",creal(r[j * n + i]), cimag(r[j*n+i]));
        }
        printf("\n");
    }*/

    info = LAPACKE_zungqr(LAPACK_COL_MAJOR, m, n, n, mat, m, tau);

    free(tau);
    if (info != 0) {
        return info;
    }
    else {
        return 0;
    }
}
