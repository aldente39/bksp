
typedef struct {
    int *I;
    int *J;
    int *nnz;
    int *row_size;
    int *col_size;
    double *value;
    char *type;
} dspmat;

typedef struct {
    int row_size;
    int col_size;
    double *value;
} dmat;

typedef struct {
    dmat *residual;
    dmat *solution;
} dres;

dspmat read_matrix_market(char *filename);
int coo2csr(dspmat *mat);
int myqr(int m, int n, double *mat, double *r);
dmat make_dmat(int m, int n);
int free_dmat(dmat *mat);
int dvec2mat(double *x, int m, dmat *mat);

/*int dcsrmm0(char type, int m, int n, double *A, int *Ai, int *Aj,
                int nnz, double *B, double *C);
 */

int dbicg(dspmat *mat, double *b,
                double tol, int max_iter, double *x);
int dbicgstab(dspmat *mat, double *b,
                double tol, int max_iter, double *x);
int dbicgstabl(dspmat *A, double *b, int l,
                double tol, int max_iter, double *x);

int dblbicgstab(dspmat *A, dmat *B,
                double tol, int max_iter, dmat *X);
int dblbicgstabl(dspmat *A, dmat *B, int l,
                double tol, int max_iter, dmat *X);

