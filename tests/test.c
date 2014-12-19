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
    dspmat *mat = read_matrix_market(f);
    printf("fnorm is %f\n", sp_norm_f(mat));
    printf("size: %d, %d \ntype: %s\n",
           *mat->row_size, *mat->col_size, mat->format);
    printf("fnorm is %f\n", sp_norm_f(mat));
    printf("size: %d, %d \ntype: %s\n",
           *mat->row_size, *mat->col_size, mat->type);fflush(stdout);

    int i, j;
    const int n = dspmat_rowsize(mat);
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
    dmat *X = dmat_create(n, s);

    int code;
    clock_t start, end;
    
    char f2[] = "../testmat/bfwb398.mtx";
    dspmat *m2 = read_matrix_market(f2);
    printf("%e, %s\n", sp_norm_f(m2), m2->type);
    double *b = (double *)calloc(n, sizeof(double));
    b[0] = 1;
    double *x2 = (double *)calloc(*m2->row_size, sizeof(double));


    // A test for the CG method.
    start = clock();
    code = bksp_dcg(m2, b, 1.0e-14, 500, x2);
    end = clock();
    printf("Computation time ... %.2f sec.\n",
           (double)(end-start)/CLOCKS_PER_SEC);
    printf("%f\n", cblas_dnrm2(*m2->row_size, x2, 1));
    if (code < 0) {
        printf("The CG Method did not converged.\n");
    }
    else {
        printf("The CG Method converged at iteration %d.\n", code);
    }
    printf("The true residual norm : %e\n",
           norm_true_res_symmetric(m2, b, x2));
    printf("\n");


    // A test for the CR method.
    for (i = 0; i < *m2->row_size; i++) {
        x2[i] = 0.0;
    }
    start = clock();
    code = dcr(m2, b, 1.0e-14, 500, x2);
    end = clock();
    printf("Computation time ... %.2f sec.\n",
           (double)(end-start)/CLOCKS_PER_SEC);
    printf("%f\n", cblas_dnrm2(*m2->row_size, x2, 1));
    if (code < 0) {
        printf("The CR Method did not converged.\n");
    }
    else {
        printf("The CR Method converged at iteration %d.\n", code);
    }
    printf("The true residual norm : %e\n",
           norm_true_res_symmetric(m2, b, x2));
    printf("\n");


    // A test for the Block CG method.
    int vn = 16;
    dmat *B_CG = dmat_create(*m2->row_size, vn);
    dmat *X_CG = dmat_create(*m2->row_size, vn);
    for (i = 0; i < vn; i++) {
        dmat_set(B_CG, i, i, 1.0);
    }
    start = clock();
    code = dblcg(m2, B_CG, 1.0e-14, 500, X_CG);
    end = clock();
    printf("Computation time ... %.2f sec.\n",
            (double)(end-start)/CLOCKS_PER_SEC);
    if (code < 0) {
        printf("The Block CG Method did not converged.\n");
    }
    else {
        printf("The Block CG Method converged at iteration %d.\n", code);
    }
    printf("\n");
    dmat_free(B_CG);
    dmat_free(X_CG);
    

    // A test for the BiCG method.
    double *xx = (double *)calloc(n, sizeof(double));
    start = clock();
    code = dbicg(mat, b, 1.0e-14, 500, xx);
    end = clock();
    printf("Computation time ... %.2f sec.\n",
           (double)(end-start)/CLOCKS_PER_SEC);
    printf("%f\n", cblas_dnrm2(n, xx, 1));
    if (code < 0) {
        printf("The BiCG Method did not converged.\n");
    }
    else {
        printf("The BiCG Method converged at iteration %d.\n", code);
    }
    printf("The true residual norm : %e\n", norm_true_res(mat, b, xx));
    printf("\n");
   

    // A test for the BiCGSTAB test.
    for (i = 0; i < n; i++) {
        xx[i] = 0.0;
    }
    start = clock();
    code = dbicgstab(mat, b, 1.0e-14, 500, xx);
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
    

    // A test for the BiCGSTAB(l) method.
    for (i = 0; i < n; i++) {
        xx[i] = 0.0;
    }
    start = clock();
    code = dbicgstabl(mat, b, 4, 1.0e-14, 500, xx);
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
    

    // A test for the Block BiCGSTAB method.
    start = clock();
    code = dblbicgstab(mat, &B, 1.0e-14, 500, X);
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
    

    // A test for the Block BiCGSTAB(l) method.
    dmat *XX = dmat_create(n, s);
    start = clock();
    code = dblbicgstabl(mat, &B, 6, 1.0e-14, 500, XX);
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


    // A test for the Shifted BiCG method.
    double sigma[4] = {0, 0.2, 0.4, 0.8};
    int sigma_num = 4;
    double *sx = (double *)calloc(n * (sigma_num + 1), sizeof(double));
    start = clock();
    code = dshbicg(mat, b, sigma, sigma_num, 1.0e-14, 500, sx);
    end = clock();
    printf("Computation time ... %.2f sec.\n",
           (double)(end-start)/CLOCKS_PER_SEC);
    printf("2-norm at the seed system : %e\n", norm_true_res(mat, b, &sx[0]));
    for (i = 0; i < sigma_num; i++) {
        printf("2-norm at sigma = %f : %e\n",
                sigma[i], norm_true_res_shift(mat, b, sigma[i], &sx[(i+1) * n]));
    }
    if (code < 0) {
        printf("The Shifted BiCG Method did not converged.\n");
    }
    else {
        printf("The Shifted BiCG Method converged.\n");
    }
    printf("\n");


    // A test for the Shifted BiCGSTAB(l) method.
    //double sigma[4] = {0, 0.2, 0.4, 0.8};
    //int sigma_num = 4;
    free(sx);
    sx = (double *)calloc(n * (sigma_num + 1), sizeof(double));
    start = clock();
    code = dshbicgstabl(mat, b, 1, sigma, sigma_num, 1.0e-14, 200, sx);
    end = clock();
    printf("Computation time ... %.2f sec.\n",
           (double)(end-start)/CLOCKS_PER_SEC);
    printf("2-norm at the seed system : %e\n", norm_true_res(mat, b, &sx[0]));
    for (i = 0; i < sigma_num; i++) {
        printf("2-norm at sigma = %f : %e\n",
                sigma[i], norm_true_res_shift(mat, b, sigma[i], &sx[(i+1) * n]));
    }
    if (code < 0) {
        printf("The Shifted BiCGSTAB(l) Method did not converged.\n");
    }
    else {
        printf("The Shifted BiCGSTAB(l) Method converged at iteration %d.\n", code);
    }
    printf("\n");

    /*
    double ***aaa = malloc(3 * sizeof(double));
    double **aa = malloc(3 * 4 * sizeof(double *));
    double *a = malloc(4 * 12 * sizeof(double));
    for(i=0;i<12;i++){
        aa[i] = &a[4 * i];
    }
    for (i =0;i<48;i++){
        a[i] = i+1;
    }
    for(i=0;i<3;i++){
        aaa[i] = &aa[i * 4];
    }
    for(i=0;i<4;i++){
        for(j=0;j<4;j++){
            printf("%f ", aaa[0][j][i]);
        }
        printf("\n");
    }*/

    return 0;
}

