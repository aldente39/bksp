#include <stdio.h>
#include <complex.h>
#include <bksp.h>
#include <mex.h>
#include "bksp_matlab.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
                      const mxArray *prhs[]) {

    // check input arguments.
    if (nrhs != 4) {
        mexErrMsgIdAndTxt("BKSP:bicg",
                          "Four inputs required.");
    }

    check_coefficient_and_rhs(prhs[0], prhs[1], "single");

    check_tolerance(prhs[2]);
    check_max_iteration(prhs[3]);

    mwIndex *row, *col;
    int ell, max_iter, i, code;
    double tol;

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
        tol = *mxGetPr(prhs[2]);
        max_iter = (int)*mxGetPr(prhs[3]);

        ////////// execute the BiCG method. //////////
        code = dbicg(mat, b, tol, max_iter, x);

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
        tol = *mxGetPr(prhs[2]);
        max_iter = (int)*mxGetPr(prhs[3]);

        if (b_im == NULL) {
            b_im = (double *)calloc(size, sizeof(double));
        }
        for(i = 0; i < size; i++) {
            b[i] = b_re[i] + b_im[i] * _Complex_I;
            x[i] = x_re[i] + x_im[i] * _Complex_I;
        }

        ////////// execute the BiCG method. //////////
        code = zbicg(mat, b, tol, max_iter, x);

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
        printf("The BiCG Method did not converged.\n");
    }
    else {
        printf("The BiCG Method converged at iteration %d.\n", code);
    }
}

