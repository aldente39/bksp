
////////// ERROR CODE //////////

#define NOT_CONVERGED -1;
#define DIVERGENCE -2;
#define INVALID_PARAMETERS -3;
#define INVALID_MATRIX_TYPE -4;
//typedef int ERROR_CODE;

////////// Structs //////////

typedef struct {
    int *I; /// Information about row
    int *J; /// Information about column
    int *nnz;
    int *row_size;
    int *col_size;
    double *value;
    char *format;
    char *type;
} dspmat;

typedef struct {
    int row_size;
    int col_size;
    double *value;
} dmat;



static inline int dmat_set (int m, int n, dmat *mat, double num) {
    mat->value[n * mat->row_size + m] = num;

    return 0;
}

static inline double dmat_get (int m, int n, dmat *mat) {
    return mat->value[n * mat->row_size + m];
}

////////// Utilities //////////

dspmat read_matrix_market(char *filename);
//int coo2csr(dspmat *mat);
int myqr(int m, int n, double *mat, double *r);
dmat *dmat_create(int m, int n);
int darray2dmat(double *arr, int row_size, int col_size, dmat *mat);
int dmat_free(dmat *mat);
int dvec2mat(double *x, int m, dmat *mat);
static inline int dmat_set(int m, int n, dmat *mat, double num);
static inline double dmat_get(int m, int n, dmat *mat);

double sp_norm_f(dspmat *A);


////////// Krylov subspace methods //////////

int bksp_dcg(dspmat *A, double *b,
             double tol, int max_iter, double *x);
int dcr(dspmat *A, double *b,
        double tol, int max_iter, double *x);

int dbicg(dspmat *A, double *b,
          double tol, int max_iter, double *x);
int dbicgstab(dspmat *A, double *b,
              double tol, int max_iter, double *x);
int dbicgstabl(dspmat *A, double *b, int l,
               double tol, int max_iter, double *x);
// TODO dcgs, dgmres


////////// Block Krylov subspace methods //////////

int dblcg(dspmat *A, dmat *B,
          double tol, int max_iter, dmat *X);

int dblbicgstab(dspmat *A, dmat *B,
                double tol, int max_iter, dmat *X);
int dblbicgstabl(dspmat *A, dmat *B, int l,
                 double tol, int max_iter, dmat *X);
// TODO dblbicg, dblcg, dblbicggr, dblgmres


////////// Krylov subspace methods for shifted systems //////////

int dshbicg(dspmat *A, double *b, double *sigma, int sigma_size,
            double tol, int max_iter, double *x);
//TODO dshbicgstabl 
int dshbicgstabl(dspmat *A, double *b, int l, double *sigma,
                 int sigma_size, double tol, int max_iter, double *x);


////////// Block Krylov subspace methods for shifted systems //////////

//TODO

