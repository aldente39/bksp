#include <stdio.h>
#include "bksp.h"
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

    ///// Initialization
    int ell, max_iter, vec_num;
    double *B_base, *X_base, tol;
    dmat *B, *X;
    dspmat *mat;
    mat = (dspmat *)malloc(sizeof(dspmat));
    B = (dmat *)malloc(sizeof(dmat));
    X = (dmat *)malloc(sizeof(dmat));

    MATLABMatrix_to_CSRMatrix(prhs[0], mat);
    vec_num = (int)mxGetN(prhs[1]);
    *plhs = mxCreateDoubleMatrix(dspmat_rowsize(mat) * vec_num, 1, mxREAL);
    mxSetM(*plhs, dspmat_rowsize(mat));
    mxSetN(*plhs, vec_num);
    X_base = mxGetPr(*plhs);
    B_base = mxGetPr(prhs[1]);
    ell = *mxGetPr(prhs[2]);
    tol = *mxGetPr(prhs[3]);
    max_iter = (int)*mxGetPr(prhs[4]);
    darray2dmat(B_base, dspmat_rowsize(mat), vec_num, B);
    darray2dmat(X_base, dspmat_rowsize(mat), vec_num, X);
    
    // execute the Block BiCGSTAB(ell) method.
    int code = dblbicgstabl(mat, B, ell, tol, max_iter, X);
    
    if (code < 0) {
        printf("The Block BiCGSTAB(l) Method did not converged.\n");
    }
    else {
        printf("The Block BiCGSTAB(l) Method converged at iteration %d.\n",
               code);
    }
    //printf("The true relative residual 2-norm : %e\n",
    //       norm_true_res_u_B(mat, B, X));

    free(X);
    free(B);
    free(mat);
}

