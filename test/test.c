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
    for (i = 0; i < n; i++) {
        xx[i] = 0.0;
    }

    int code;
    clock_t start, end;
    
    start = clock();
    code = dbicg(&mat, bb, 1.0e-14, 500, xx);
    end = clock();
    printf("Computation time ... %.2f sec.\n",
           (double)(end-start)/CLOCKS_PER_SEC);
    printf("%f\n", cblas_dnrm2(n, xx, 1));
    if (code < 0) {
        printf("The BiCG Method did not converged.\n");
    }
    else {
        printf("The BiCG Method converged.\n");
    }
    printf("\n");
   
    for (i = 0; i < n; i++) {
        xx[i] = 0.0;
    }
    start = clock();
    code = dbicgstab(&mat, bb, 1.0e-14, 500, xx);
    end = clock();
    printf("Computation time ... %.2f sec.\n",
           (double)(end-start)/CLOCKS_PER_SEC);
    printf("%f\n", cblas_dnrm2(n, xx, 1));
    if (code < 0) {
        printf("The BiCGSTAB Method did not converged.\n");
    }
    else {
        printf("The BiCGSTAB Method converged.\n");
    }
    printf("\n");
    
    for (i = 0; i < n; i++) {
        xx[i] = 0.0;
    }
    start = clock();
    code = dbicgstabl(&mat, bb, 4, 1.0e-14, 500, xx);
    end = clock();
    printf("Computation time ... %.2f sec.\n",
           (double)(end-start)/CLOCKS_PER_SEC);
    printf("%f\n", cblas_dnrm2(n, xx, 1));
    if (code < 0) {
        printf("The BiCGSTAB(l) Method did not converged.\n");
    }
    else {
        printf("The BiCGSTAB(l) Method converged.\n");
    }
    printf("\n");
    
    start = clock();
    code = dblbicgstab(&mat, &B, 1.0e-14, 500, &X);
    end = clock();
    printf("Computation time ... %.2f sec.\n",
            (double)(end-start)/CLOCKS_PER_SEC);
    if (code < 0) {
        printf("The Block BiCGSTAB Method did not converged.\n");
    }
    else {
        printf("The Block BiCGSTAB Method converged.\n");
    }
    printf("\n");
    
    dmat XX = make_dmat(n, s);
    start = clock();
    code = dblbicgstabl(&mat, &B, 6, 1.0e-14, 500, &XX);
    end = clock();
    printf("Computation time ... %.2f sec.\n",
            (double)(end-start)/CLOCKS_PER_SEC);
    if (code < 0) {
        printf("The Block BiCGSTAB(l) Method did not converged.\n");
    }
    else {
        printf("The Block BiCGSTAB(l) Method converged.\n");
    }
    printf("\n");


    return 0;
}

