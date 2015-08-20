#include <stdio.h>
#include <complex.h>
#include <bksp.h>
#include <mex.h>
#include "bksp_matlab.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
                      const mxArray *prhs[]) {

    // check input arguments.
    if (nrhs != 5) {
        mexErrMsgIdAndTxt("BKSP:bicgstabl",
                          "Five inputs required.");
    }

    check_coefficient_and_rhs(prhs[0], prhs[1], "single");

    if (mxGetM(prhs[2]) != 1 || mxGetN(prhs[2]) != 1) {
        mexErrMsgIdAndTxt("BKSP:bicgstabl",
                          "Input parameter ell must be an integer.");
    }

    check_tolerance(prhs[3]);
    check_max_iteration(prhs[4]);

    ////////// Initialization //////////
    mwIndex *row, *col;
    int ell, max_iter, i, code;
    double tol;

    ell = *mxGetPr(prhs[2]);
    tol = *mxGetPr(prhs[3]);
    max_iter = (int)*mxGetPr(prhs[4]);

    if (mxGetPi(prhs[0]) == NULL) {
        ////////// Initialization //////////
        dspmat *mat;
        double *b, *x;

        // convert MATLAB sparse matrix to CSR matrix.
        mat = (dspmat *)malloc(sizeof(dspmat));
        MATLABMatrix2dCSRMatrix(prhs[0], mat);

        *plhs = mxCreateDoubleMatrix(dspmat_rowsize(mat), 1, mxREAL);
        x = mxGetPr(*plhs);
        b = mxGetPr(prhs[1]);

        ////////// execute the BiCGSTAB(l) method. //////////
        code = dbicgstabl(mat, b, ell, tol, max_iter, x);

        ////////// Finalization //////////
        free(mat);
    }
    else if (mxIsComplex(prhs[0])) {
        ////////// Initialization //////////
        zspmat *mat;
        double _Complex *b, *x;
        double *b_re, *b_im, *x_re, *x_im;

        // convert MATLAB sparse matrix to CSR matrix.        
        mat = (zspmat *)malloc(sizeof(zspmat));
        MATLABMatrix2zCSRMatrix(prhs[0], mat);

        int size = zspmat_rowsize(mat);
        b = (double _Complex *)malloc(sizeof(double _Complex) * size);
        x = (double _Complex *)malloc(sizeof(double _Complex) * size);

        *plhs = mxCreateDoubleMatrix(size, 1, mxCOMPLEX);
        x_re = mxGetPr(*plhs);
        x_im = mxGetPi(*plhs);
        b_re = mxGetPr(prhs[1]);
        b_im = mxGetPi(prhs[1]);

        if (b_im == NULL) {
            b_im = (double *)calloc(size, sizeof(double));
        }
        for(i = 0; i < size; i++) {
            b[i] = b_re[i] + b_im[i] * _Complex_I;
            x[i] = x_re[i] + x_im[i] * _Complex_I;
        }

        ////////// execute the BiCGSTAB(l) method. //////////
        code = zbicgstabl(mat, b, ell, tol, max_iter, x);

        ////////// Finalization //////////
        // store approximate solution for MATLAB.
        for (i = 0; i < size; i++) {
            x_re[i] = creal(x[i]);
            x_im[i] = cimag(x[i]);
        }
        mxSetPr(*plhs, x_re);
        mxSetPi(*plhs, x_im);

        free(mat);
        free(b);
        free(x);
    }

    if (code < 0) {
        printf("The BiCGSTAB(l) Method did not converged.\n");
    }
    else {
        printf("The BiCGSTAB(l) Method converged at iteration %d.\n", code);
    }
}

