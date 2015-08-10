#include <string.h>
#include <mex.h>
#include "bksp.h"

void MATLABMatrix_to_CSRMatrix (const mxArray *MATLABmat, dspmat *mat) {

    mwIndex *row, *col;
    int size, i;
    mxArray *spMat;
    mexCallMATLAB(1, &spMat,
                  1, (mxArray **)&MATLABmat, "transpose");

    size = (int)mxGetM(spMat);
    mat->nnz = (int *)(mxGetJc(spMat) + mxGetN(spMat));
    row = mxGetJc(spMat);
    col = mxGetIr(spMat);
    mat->row = (int *)malloc(sizeof(int) * (size + 1));
    mat->col = (int *)malloc(sizeof(int) * (*mat->nnz));

    mat->row_size = (int *)malloc(sizeof(int));
    mat->col_size = (int *)malloc(sizeof(int));
    *mat->row_size = size;
    *mat->col_size = size;
    mat->value = mxGetPr(spMat);
    mat->format = "CRS";

    for (i = 0; i < *mat->nnz; i++) {
        mat->col[i] = (int)col[i];
    }
    for (i = 0; i <= size; i++) {
        mat->row[i] = (int)row[i];
    }
}

void check_coefficient_and_rhs (const mxArray *A, const mxArray *b,
                                char *type) {

    int row_size, col_size, vec_size, vec_num;
    row_size = mxGetM(A);
    col_size = mxGetN(A);
    vec_size = mxGetM(b);
    vec_num = mxGetN(b);

    if (!mxIsSparse(A) || row_size != col_size) {
        mexErrMsgIdAndTxt( "BKSP:invalidInputSparisty",
                           "Input argument must be a sparse square matrix.");
    }

    if (!strcmp(type, "single")) {
        if (vec_num != 1) {
            mexErrMsgIdAndTxt( "BKSP:invalidInputVector",
                               "Input right hand side must be a vector.");
        }
    }
    else if (!strcmp(type, "block")) {
    }

    if (vec_size != row_size) {
        mexErrMsgIdAndTxt( "BKSP:invalidInputVector",
                           "Input right hand side must have the same size of the coefficient matrix.");
    }
}

void check_tolerance (const mxArray *tol) {

    if (mxGetM(tol) != 1 || mxGetN(tol) != 1) {
        mexErrMsgIdAndTxt("BKSP:blbicgstabl",
                          "Input tolerance must be a scalar.");
    }
}

void check_max_iteration (const mxArray *maxit) {

    if (mxGetM(maxit) != 1 || mxGetN(maxit) != 1) {
        mexErrMsgIdAndTxt("BKSP:blbicgstabl",
                          "Input max iteration must be an integer.");
    }
}

