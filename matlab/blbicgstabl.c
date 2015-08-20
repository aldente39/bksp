#include <stdio.h>
#include <complex.h>
#include <bksp.h>
#include <mex.h>
#include "bksp_matlab.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
                      const mxArray *prhs[]) {

    ///// check input arguments.
    if (nrhs != 5) {
        mexErrMsgIdAndTxt("BKSP:blbicgstabl",
                          "Five inputs required.");
    }

    check_coefficient_and_rhs(prhs[0], prhs[1], "block");

    if (mxGetM(prhs[2]) != 1 || mxGetN(prhs[2]) != 1) {
        mexErrMsgIdAndTxt("BKSP:blbicgstabl",
                          "Input parameter ell must be an integer.");
    }

    check_tolerance(prhs[3]);
    check_max_iteration(prhs[4]);

    ////////// Initialization //////////
    mwIndex *row, *col;
    int ell, max_iter, i, code, vec_num;
    double tol;

    ell = *mxGetPr(prhs[2]);
    tol = *mxGetPr(prhs[3]);
    max_iter = (int)*mxGetPr(prhs[4]);

    if (mxGetPi(prhs[0]) == NULL) {
        ////////// Initialization //////////
        dspmat *mat;
        dmat *B, *X;
        double *B_base, *X_base;
        mat = (dspmat *)malloc(sizeof(dspmat));
        B = (dmat *)malloc(sizeof(dmat));
        X = (dmat *)malloc(sizeof(dmat));

        // convert MATLAB sparse matrix to CSR matrix.
        mat = (dspmat *)malloc(sizeof(dspmat));
        MATLABMatrix2dCSRMatrix(prhs[0], mat);
        vec_num = (int)mxGetN(prhs[1]);
        *plhs = mxCreateDoubleMatrix(dspmat_rowsize(mat) * vec_num, 1, mxREAL);
        mxSetM(*plhs, dspmat_rowsize(mat));
        mxSetN(*plhs, vec_num);
        X_base = mxGetPr(*plhs);
        B_base = mxGetPr(prhs[1]);

        darray2dmat(B_base, dspmat_rowsize(mat), vec_num, B);
        darray2dmat(X_base, dspmat_rowsize(mat), vec_num, X);

        ////////// execute the Block BiCGSTAB(l) method. //////////
        code = dblbicgstabl(mat, B, ell, tol, max_iter, X);

        ////////// Finalization //////////
        free(X);
        free(B);
        free(mat);
    }
    else if (mxIsComplex(prhs[0])) {
        ////////// Initialization //////////
        zspmat *mat;
        zmat *B, *X;
        double *B_re, *B_im, *X_re, *X_im;
        double _Complex *B_base, *X_base;
        B = (zmat *)malloc(sizeof(zmat));
        X = (zmat *)malloc(sizeof(zmat));

        // convert MATLAB sparse matrix to CSR matrix.        
        mat = (zspmat *)malloc(sizeof(zspmat));
        MATLABMatrix2zCSRMatrix(prhs[0], mat);
        vec_num = (int)mxGetN(prhs[1]);
        int elem_size = zspmat_rowsize(mat) * vec_num;
        B_base = (double _Complex *)malloc(sizeof(double _Complex) * elem_size);
        X_base = (double _Complex *)malloc(sizeof(double _Complex) * elem_size);

        *plhs = mxCreateDoubleMatrix(elem_size, 1, mxCOMPLEX);
        mxSetM(*plhs, zspmat_rowsize(mat));
        mxSetN(*plhs, vec_num);
        X_re = mxGetPr(*plhs);
        X_im = mxGetPi(*plhs);
        B_re = mxGetPr(prhs[1]);
        B_im = mxGetPi(prhs[1]);

        if (B_im == NULL) {
            B_im = (double *)calloc(elem_size, sizeof(double));
        }
        for(i = 0; i < elem_size; i++) {
            B_base[i] = B_re[i] + B_im[i] * _Complex_I;
            X_base[i] = X_re[i] + X_im[i] * _Complex_I;
        }

        B->value = B_base;
        B->row_size = zspmat_rowsize(mat);
        B->col_size = vec_num;
        X->value = X_base;
        X->row_size = zspmat_rowsize(mat);
        X->col_size = vec_num;

        ////////// execute the BiCG method. //////////
        code = zblbicgstabl(mat, B, ell, tol, max_iter, X);

        ////////// Finalization //////////
        // store approximate solution for MATLAB.
        for (i = 0; i < elem_size; i++) {
            X_re[i] = creal(X->value[i]);
            X_im[i] = cimag(X->value[i]);
        }
        mxSetPr(*plhs, X_re);
        mxSetPi(*plhs, X_im);

        free(mat);
        zmat_free(B);
        zmat_free(X);
    }

    if (code < 0) {
        printf("The Block BiCGSTAB(l) Method did not converged.\n");
    }
    else {
        printf("The Block BiCGSTAB(l) Method converged at iteration %d.\n",
               code);
    }
}

