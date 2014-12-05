#include <stdio.h>
#include <math.h>
#include "bksp.h"
#include <mkl.h>
#include <time.h>
#include <string.h>

double norm_true_res(dspmat *A, double *b, double *x) {
    double ans;
    int n = *(A->row_size);
    double *tmp = (double *) malloc(n * sizeof(double));
    mkl_cspblas_dcsrgemv("n", A->row_size, A->value,
                         A->I, A->J, x, tmp);
    cblas_dscal(n, -1, tmp, 1);
    cblas_daxpy(n, 1, b, 1, tmp, 1);
    ans = cblas_dnrm2(n, tmp, 1) / cblas_dnrm2(n, b, 1);
    
    free(tmp);
    return ans;
}

double norm_true_res_symmetric(dspmat *A, double *b, double *x) {
    double ans;
    int n = *(A->row_size);
    double *tmp = (double *) malloc(n * sizeof(double));
    mkl_cspblas_dcsrsymv("l", A->row_size, A->value,
                         A->I, A->J, x, tmp);
    cblas_dscal(n, -1, tmp, 1);
    cblas_daxpy(n, 1, b, 1, tmp, 1);
    ans = cblas_dnrm2(n, tmp, 1) / cblas_dnrm2(n, b, 1);
    
    free(tmp);
    return ans;
}

double norm_true_res_shift(dspmat *A, double *b, double sigma, double *x) {
    double ans;
    int n = *(A->row_size);
    double *tmp = (double *) malloc(n * sizeof(double));
    mkl_cspblas_dcsrgemv("n", A->row_size, A->value,
                         A->I, A->J, x, tmp);
    cblas_dscal(n, -1, tmp, 1);
    cblas_daxpy(n, 1, b, 1, tmp, 1);
    cblas_daxpy(n, - sigma, x, 1, tmp, 1);
    ans = cblas_dnrm2(n, tmp, 1) / cblas_dnrm2(n, b, 1);
    
    free(tmp);
    return ans;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("input matrix file name.\n");
        return -1;
    }

    char f[] = "";
    strcat(f, argv[1]);
    dspmat mat;
    mat = read_matrix_market(f);
    printf("fnorm is %f\n", sp_norm_f(&mat));
    printf("size: %d, %d \ntype: %s\n",
           *mat.row_size, *mat.col_size, mat.format);
    printf("fnorm is %f\n", sp_norm_f(&mat));
    printf("size: %d, %d \ntype: %s\n",
           *mat.row_size, *mat.col_size, mat.type);

    int i, j;
    const int n = *mat.row_size;
    const int s = 16;
    double *bb = (double *)calloc(sizeof(double), n * s * sizeof(double));
    int seed[4] = {1,0,3,2};
    LAPACKE_dlarnv(1, seed, n * s, bb);
    for (i = 0; i < s; i++) {
        bb[(n + 1) * i] = 1;
    }
    dmat B;
    //B = make_dmat(n, s);
    //copy_darray2mat('c', bb, &B);
    darray2dmat(bb, n, s, &B);
    dmat X = make_dmat(n, s);
    double *xx = malloc(n * sizeof(double));
    for (i = 0; i < n; i++) {
        xx[i] = 0.0;
    }

    int code;
    clock_t start, end;
    
    char f2[] = "../testmat/crystm01.mtx";
    dspmat m2 = read_matrix_market(f2);
    printf("%e, %s\n", sp_norm_f(&m2), m2.type);
    double *b = (double *)calloc(*m2.row_size, sizeof(double));
    b[0] = 1;
    double *x2 = (double *)calloc(*m2.row_size, sizeof(double));


    start = clock();
    code = bksp_dcg(&m2, b, 1.0e-14, 500, x2);
    end = clock();
    printf("Computation time ... %.2f sec.\n",
           (double)(end-start)/CLOCKS_PER_SEC);
    printf("%f\n", cblas_dnrm2(*m2.row_size, x2, 1));
    if (code < 0) {
        printf("The CG Method did not converged.\n");
    }
    else {
        printf("The CG Method converged at iteration %d.\n", code);
    }
    printf("The true residual norm : %e\n",
           norm_true_res_symmetric(&m2, b, x2));
    printf("\n");


    for (i = 0; i < n; i++) {
        x2[i] = 0.0;
    }
    start = clock();
    code = dcr(&m2, b, 1.0e-14, 500, x2);
    end = clock();
    printf("Computation time ... %.2f sec.\n",
           (double)(end-start)/CLOCKS_PER_SEC);
    printf("%f\n", cblas_dnrm2(*m2.row_size, x2, 1));
    if (code < 0) {
        printf("The CR Method did not converged.\n");
    }
    else {
        printf("The CR Method converged at iteration %d.\n", code);
    }
    printf("The true residual norm : %e\n",
           norm_true_res_symmetric(&m2, b, x2));
    printf("\n");
    

    for (i = 0; i < n; i++) {
        xx[i] = 0.0;
    }
    start = clock();
    code = dbicg(&mat, b, 1.0e-14, 500, xx);
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
    printf("The true residual norm : %e\n", norm_true_res(&mat, b, xx));
    printf("\n");
   

    for (i = 0; i < n; i++) {
        xx[i] = 0.0;
    }
    start = clock();
    code = dbicgstab(&mat, b, 1.0e-14, 500, xx);
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
    code = dbicgstabl(&mat, b, 4, 1.0e-14, 500, xx);
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


    double sigma[4] = {0, 0.2, 0.4, 0.8};
    int sigma_num = 4;
    double *sx = (double *)calloc(n * (sigma_num + 1), sizeof(double));
    start = clock();
    code = dshbicg(&mat, b, sigma, sigma_num, 1.0e-14, 500, sx);
    end = clock();
    printf("Computation time ... %.2f sec.\n",
           (double)(end-start)/CLOCKS_PER_SEC);
    printf("2-norm at the seed system : %e\n", norm_true_res(&mat, b, &sx[0]));
    for (i = 0; i < sigma_num; i++) {
        printf("2-norm at sigma = %f : %e\n",
                sigma[i], norm_true_res_shift(&mat, b, sigma[i], &sx[(i+1) * n]));
    }
    if (code < 0) {
        printf("The Shifted BiCG Method did not converged.\n");
    }
    else {
        printf("The Shifted BiCG Method converged.\n");
    }
    printf("\n");

    return 0;
}

