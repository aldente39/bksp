#include <stdio.h>
#include <math.h>
#include "bksp.h"
#include <mkl.h>
#include <time.h>
#include <string.h>

double fnorm(dspmat m) {
    double n = 0;
    int i;
    int end = *m.nnz;
    for(i = 0; i < end; i++) {
        n += m.value[i] * m.value[i];
    }
    return sqrt(n);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("input matrix file name.\n");
        return -1;
    }

    char f[] = "";
    strcat(f, argv[1]);
    dspmat mat;
    int ret;
    mat = read_matrix_market(f);
    printf("fnorm is %f\n", fnorm(mat));
    printf("size: %d, %d \ntype: %s\n",
           *mat.row_size, *mat.col_size, mat.type);

    ret = coo2csr(&mat);

    printf("%d\n",ret);
    printf("fnorm is %f\n", fnorm(mat));
    printf("size: %d, %d \ntype: %s\n",
           *mat.row_size, *mat.col_size, mat.type);

    int i;
    const int n = *mat.row_size;
    const int s = 16;
    double *bb = (double *)calloc(sizeof(double), n * s * sizeof(double));
    int seed[4] = {1,0,3,2};
    LAPACKE_dlarnv(1, seed, n * s, bb);
    for (i = 0; i < s; i++) {
        bb[(n + 1) * i] = 1;
    }
    dmat B;
    B = make_dmat(n, s);
    copy_darray2mat('c', bb, &B);
    dmat X = make_dmat(n, s);
    double *xx = malloc(n * sizeof(double));

    clock_t start, end;
    start = clock();
    dblbicgstab(&mat, &B, 4, 1.0e-14, 500, &X);
    end = clock();
    printf("Computation time ... %.2f sec.\n",
            (double)(end-start)/CLOCKS_PER_SEC);
    dmat XX = make_dmat(n, s);
    start = clock();
    dblbicgstabl(&mat, &B, 6, 1.0e-14, 500, &XX);
    end = clock();
    printf("Computation time ... %.2f sec.\n",
            (double)(end-start)/CLOCKS_PER_SEC);


    char *matdescra = "GLNF";
    int m = *mat.row_size;
    int L = 4;
    double *tmp = (double *)malloc(m * L * sizeof(double));
    double *g = (double *)calloc(sizeof(double),m * L * sizeof(double));
    for (i = 0; i < L; i++) {
        g[(m + 1) * i] = 1;
    }

    double one = 1;
    double zero = 0;
    int *row_ptr = malloc((m + 1) * sizeof(int));
    for (i = 0; i < m + 1; i++) {
        row_ptr[i] = mat.I[i] + 1;
    }
    int *col_ind = malloc(*mat.nnz * sizeof(int));
    for (i = 0; i < *mat.nnz; i++) {
        col_ind[i] = mat.J[i] + 1;
    }

    return 0;
}

