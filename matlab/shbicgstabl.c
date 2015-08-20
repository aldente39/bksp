#include <stdio.h>
#include <complex.h>
#include <bksp.h>
#include <mex.h>
#include "bksp_matlab.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
                      const mxArray *prhs[]) {

    // check input arguments.
    if (nrhs != 6) {
        mexErrMsgIdAndTxt("BKSP:shbicgstabl",
                          "Six inputs required.");
    }

    check_coefficient_and_rhs(prhs[0], prhs[1], "single");

    if (mxGetM(prhs[2]) != 1 || mxGetN(prhs[2]) != 1) {
        mexErrMsgIdAndTxt("BKSP:shbicgstabl",
                          "Input parameter ell must be an integer.");
    }

    check_tolerance(prhs[4]);
    check_max_iteration(prhs[5]);

    ////////// Initialization //////////
    mwIndex *row, *col;
    int ell, max_iter, i, code, sigmas_size, total_size;
    double tol;

    ell = *mxGetPr(prhs[2]);
    tol = *mxGetPr(prhs[4]);
    max_iter = (int)*mxGetPr(prhs[5]);
    sigmas_size = (int)((mxGetM(prhs[3]) > mxGetN(prhs[3])) ?
                       mxGetM(prhs[3]) :
                       mxGetN(prhs[3]));

    if (mxGetPi(prhs[0]) == NULL) {
        ////////// Initialization //////////
        dspmat *mat;
        double *b, *x, *sigmas;

        // convert MATLAB sparse matrix to CSR matrix.
        mat = (dspmat *)malloc(sizeof(dspmat));
        MATLABMatrix2dCSRMatrix(prhs[0], mat);

        total_size = dspmat_rowsize(mat) * (sigmas_size + 1);
        *plhs = mxCreateDoubleMatrix(total_size, 1, mxREAL);
        mxSetM(*plhs, dspmat_rowsize(mat));
        mxSetN(*plhs, sigmas_size + 1);
        x = mxGetPr(*plhs);
        b = mxGetPr(prhs[1]);
        sigmas = mxGetPr(prhs[3]);

        ////////// execute the Shifted BiCGSTAB(l) method. //////////
        code = dshbicgstabl(mat, b, ell, sigmas, sigmas_size, tol, max_iter, x);

        ////////// Finalization //////////
        free(mat);
    }
    else if (mxIsComplex(prhs[0])) {
        ////////// Initialization //////////
        zspmat *mat;
        double _Complex *b, *x, *sigmas;
        double *b_re, *b_im, *x_re, *x_im, *sigmas_re, *sigmas_im;

        // convert MATLAB sparse matrix to CSR matrix.        
        mat = (zspmat *)malloc(sizeof(zspmat));
        MATLABMatrix2zCSRMatrix(prhs[0], mat);

        int size = zspmat_rowsize(mat);
        b = (double _Complex *)malloc(sizeof(double _Complex) * size);
        x = (double _Complex *)calloc(size * (sigmas_size + 1),
                                      sizeof(double _Complex));
        sigmas = (double _Complex *)malloc(sizeof(double _Complex) * sigmas_size);

        total_size = size * (sigmas_size + 1);
        *plhs = mxCreateDoubleMatrix(total_size, 1, mxCOMPLEX);
        mxSetM(*plhs, size);
        mxSetN(*plhs, sigmas_size + 1);
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

        sigmas_re = mxGetPr(prhs[3]);
        sigmas_im = mxGetPi(prhs[3]);
        if (sigmas_im == NULL) {
            sigmas_im = (double *)calloc(sigmas_size, sizeof(double));
        }
        for(i = 0; i < sigmas_size; i++) {
            sigmas[i] = sigmas_re[i] + sigmas_im[i] * _Complex_I;
        }

        ////////// execute the Shifted BiCGSTAB(l) method. //////////
        code = zshbicgstabl(mat, b, ell, sigmas, sigmas_size, tol, max_iter, x);

        ////////// Finalization //////////
        // store approximate solution for MATLAB.
        for (i = 0; i < total_size; i++) {
            x_re[i] = creal(x[i]);
            x_im[i] = cimag(x[i]);
        }
        mxSetPr(*plhs, x_re);
        mxSetPi(*plhs, x_im);

        free(mat);
        free(b);
        free(x);
        free(sigmas);
    }

    if (code < 0) {
        printf("The Shifted BiCGSTAB(l) Method did not converged.\n");
    }
    else {
        printf("The Shifted BiCGSTAB(l) Method converged at iteration %d.\n",
               code);
    }
}

