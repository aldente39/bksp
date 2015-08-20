
////////// ERROR CODE //////////

#define NOT_CONVERGED -1;
#define DIVERGENCE -2;
#define INVALID_PARAMETERS -3;
#define INVALID_MATRIX_TYPE -4;
//typedef int ERROR_CODE;

////////// Structs //////////

typedef struct {
    int *row; /// Information about row
    int *col; /// Information about column
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

////////// Utilities //////////

static inline int dspmat_rowsize(dspmat *mat) {
    return *mat->row_size;
}

static inline int dspmat_colsize(dspmat *mat) {
    return *mat->col_size;
}

static inline int dmat_set (dmat *mat, int m, int n, double num) {
    mat->value[n * mat->row_size + m] = num;
    return 0;
}

static inline double dmat_get (dmat *mat, int m, int n) {
    return mat->value[n * mat->row_size + m];
}

static inline int dmat_rowsize (dmat *mat) {
    return mat->row_size;
}

static inline int dmat_colsize (dmat *mat) {
    return mat->col_size;
}

static inline double* dmat_value (dmat *mat) {
    return mat->value;
}

dspmat *read_matrix_market(char *filename);
//int coo2csr(dspmat *mat);
int myqr(int m, int n, double *mat, double *r);
dmat *dmat_create(int m, int n);
int darray2dmat(double *arr, int row_size, int col_size, dmat *mat);
int dmat_free(dmat *mat);
int dvec2mat(double *x, int m, dmat *mat);


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


///// For complex numbers

////////// Structs //////////

typedef struct {
    int *row; /// Information about row
    int *col; /// Information about column
    int *nnz;
    int *row_size;
    int *col_size;
    double _Complex *value;
    char *format;
    char *type;
} zspmat;

typedef struct {
    int row_size;
    int col_size;
    double _Complex *value;
} zmat;

////////// Utilities //////////

static inline int zspmat_rowsize(zspmat *mat) {
    return *mat->row_size;
}

static inline int zspmat_colsize(zspmat *mat) {
    return *mat->col_size;
}

static inline int zmat_set (zmat *mat, int m, int n, double _Complex *num) {
    mat->value[n * mat->row_size + m] = *num;
    return 0;
}

static inline double zmat_get (zmat *mat, int m, int n) {
    return mat->value[n * mat->row_size + m];
}

static inline int zmat_rowsize (zmat *mat) {
    return mat->row_size;
}

static inline int zmat_colsize (zmat *mat) {
    return mat->col_size;
}

static inline double _Complex* zmat_value (zmat *mat) {
    return mat->value;
}

zspmat *zread_matrix_market(char *filename);
int zqreco(int m, int n, double _Complex *mat, double _Complex *r);
zmat *zmat_create(int m, int n);
int zmat_free(zmat *mat);

////////// Krylov subspace methods //////////

int zbicg(zspmat *A, double _Complex *b,
          double tol, int max_iter, double _Complex *x);
int zbicgstabl(zspmat *A, double _Complex *b, int ell,
          double tol, int max_iter, double _Complex *x);

////////// Block Krylov subspace methods //////////

int zblbicgstabl(zspmat *A, zmat *b, int ell,
                 double tol, int max_iter, zmat *x);

////////// Krylov subspace methods for shifted systems //////////

int zshbicgstabl(zspmat *A, double _Complex *b, int l, double _Complex *sigma,
                 int sigma_size, double tol, int max_iter, double _Complex *x);

////////// Block Krylov subspace methods for shifted systems //////////

// TODO
