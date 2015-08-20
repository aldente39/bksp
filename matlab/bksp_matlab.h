
void MATLABMatrix2dCSRMatrix (const mxArray *MATLABmat, dspmat *mat);
void MATLABMatrix2zCSRMatrix (const mxArray *MATLABmat, zspmat *mat);
void check_coefficient_and_rhs (const mxArray *A, const mxArray *b, char *type);
void check_tolerance (const mxArray *tol);
void check_max_iteration (const mxArray *maxit);
