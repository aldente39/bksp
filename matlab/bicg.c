#include <stdio.h>
#include "bksp.h"
#include <mex.h>
#include "bksp_matlab.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
                      const mxArray *prhs[]) {

    ///// check input arguments.
    if (nrhs != 4) {
        mexErrMsgIdAndTxt("BKSP:bicg",
                          "Four inputs required.");
    }

    check_coefficient_and_rhs(prhs[0], prhs[1], "single");

    check_tolerance(prhs[2]);
    check_max_iteration(prhs[3]);
    
    ///// Initialization
    mwIndex *row, *col;
    int ell, max_iter, i;
    double *b, *x, tol;
    dspmat *mat;
    mat = (dspmat *)malloc(sizeof(dspmat));

    MATLABMatrix_to_CSRMatrix(prhs[0], mat);

    *plhs = mxCreateDoubleMatrix(dspmat_rowsize(mat), 1, mxREAL);
    x = mxGetPr(*plhs);

    b = mxGetPr(prhs[1]);
    tol = *mxGetPr(prhs[2]);
    max_iter = (int)*mxGetPr(prhs[3]);

    /* for (i = 0; i < *mat->nnz; i++) {
     *     mat->col[i] = (int)col[i];
     * }
     * for (i = 0; i <= dspmat_rowsize(mat); i++) {
     *     mat->row[i] = (int)row[i];
     * } */

    // execute the BiCG method.
    int code = dbicg(mat, b, tol, max_iter, x);

    if (code < 0) {
        printf("The BiCG Method did not converged.\n");
    }
    else {
        printf("The BiCG Method converged at iteration %d.\n", code);
    }
    printf("The true relative residual 2-norm : %e\n",
           d2norm_true_res_u(mat, b, x));

    free(mat);
}

