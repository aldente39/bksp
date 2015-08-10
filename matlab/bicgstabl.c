#include <stdio.h>
#include "bksp.h"
#include <mex.h>
#include "bksp_matlab.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
                      const mxArray *prhs[]) {

    ///// check input arguments.
    if (nrhs != 5) {
        mexErrMsgIdAndTxt("MyProg:mysqrt",
                          "Five inputs required.");
    }

    check_coefficient_and_rhs(prhs[0], prhs[1], "single");

    if (mxGetM(prhs[2]) != 1 || mxGetN(prhs[2]) != 1) {
        mexErrMsgIdAndTxt("BKSP:blbicgstabl",
                          "Input parameter ell must be an integer.");
    }

    check_tolerance(prhs[3]);
    check_max_iteration(prhs[4]);

    ///// Initialization
    int ell, max_iter, size;
    double *b, *x, tol;
    dspmat *mat;
    mat = (dspmat *)malloc(sizeof(dspmat));

    MATLABMatrix_to_CSRMatrix(prhs[0], mat);

    *plhs = mxCreateDoubleMatrix(dspmat_rowsize(mat), 1, mxREAL);
    x = mxGetPr(*plhs);
    b = mxGetPr(prhs[1]);
    ell = *mxGetPr(prhs[2]);
    tol = *mxGetPr(prhs[3]);
    max_iter = (int)*mxGetPr(prhs[4]);
    
    // execute the BiCGSTAB(ell) method.
    int code = dbicgstabl(mat, b, ell, tol, max_iter, x);

    if (code < 0) {
        printf("The BiCGSTAB(l) Method did not converged.\n");
    }
    else {
        printf("The BiCGSTAB(l) Method converged at iteration %d.\n", code);
    }
    printf("The true relative residual 2-norm : %e\n",
           d2norm_true_res_u(mat, b, x));

    free(mat);
}

