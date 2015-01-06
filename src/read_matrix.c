/* 
*   Matrix Market I/O example program
*
*   Read a real (non-complex) sparse matrix from a Matrix Market (v. 2.0) file.
*   and copies it to stdout.  This porgram does nothing useful, but
*   illustrates common usage of the Matrix Matrix I/O routines.
*   (See http://math.nist.gov/MatrixMarket for details.)
*
*   Usage:  a.out [filename] > output
*
*       
*   NOTES:
*
*   1) Matrix Market files are always 1-based, i.e. the index of the first
*      element of a matrix is (1,1), not (0,0) as in C.  ADJUST THESE
*      OFFSETS ACCORDINGLY offsets accordingly when reading and writing 
*      to files.
*
*   2) ANSI C requires one to use the "l" format modifier when reading
*      double precision floating point numbers in scanf() and
*      its variants.  For example, use "%lf", "%lg", or "%le"
*      when reading doubles, otherwise errors will occur.
*/

#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"
#include "bksp_internal.h"
#include "bksp.h"

dspmat *read_matrix_market(char *filename){
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nz;   
    int i, *row, *col;
    double *val;
    dspmat *mat;
    int *row_size, *col_size, *nnz;
    mat = (dspmat *)malloc(sizeof(dspmat));
    nnz = (int *) malloc(sizeof(int));
    row_size = (int *) malloc(sizeof(int));
    col_size = (int *) malloc(sizeof(int));

    if ((f = fopen(filename, "r")) == NULL) {
        exit(1);
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }


    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
            mm_is_sparse(matcode) )
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* find out size of sparse matrix .... */

    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
        exit(1);


    /* reseve memory for matrices */

    row = (int *) malloc(nz * sizeof(int));
    col = (int *) malloc(nz * sizeof(int));
    val = (double *) malloc(nz * sizeof(double));


    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i=0; i<nz; i++)
    {
        fscanf(f, "%d %d %lg\n", &row[i], &col[i], &val[i]);
        row[i]--;  /* adjust from 1-based to 0-based */
        col[i]--;
    }

    if (f !=stdin) fclose(f);

    *row_size = M;
    *col_size = N;
    *nnz = nz;
    mat->row = row;
    mat->col = col;
    mat->nnz = nnz;
    mat->row_size = row_size;
    mat->col_size = col_size;
    mat->value = val;
    mat->format = "coo";
    if (matcode[3] == 'S') {
        mat->type = "symmetric";
    }
    else {
        mat->type = "unsymmetric";
    }

    coo2csr(mat);

    return mat;
}

int coo2csr(dspmat *mat) {
    MKL_INT *n, *ja, *ia, *nnz, *row_ind, *col_ind, info;
    double *Acsr, *Acoo;
    MKL_INT job[] = {1,0,0,0,*mat->nnz,0};
    n = mat->row_size;
    ja = (int *)calloc((*mat->nnz), sizeof(int));
    ia = (int *)calloc((*n + 1), sizeof(int));
    Acsr = (double *)malloc((*mat->nnz) * sizeof(double));
    Acoo = mat->value;
    nnz = mat->nnz;
    row_ind = mat->row;
    col_ind = mat->col;

    /*
    mkl_dcsrcoo(job, n, Acsr, ja, ia, nnz, Acoo, row_ind, col_ind, &info);
    if(info != 0){
        return -1;
    }
    */
    
    int i, tmp1, tmp2;
    for (i = 0; i < *nnz; i++) {
        ia[row_ind[i]]++;
    }
    for (i = 0, tmp2 = 0; i < *n; i++) {
        tmp1 = ia[i];
        ia[i] = tmp2;
        tmp2 += tmp1;
    }
    ia[*n] = *nnz;

    for (i = 0; i < *nnz; i++) {
        tmp1 = row_ind[i];
        tmp2 = ia[tmp1];
        ja[tmp2] = col_ind[i];
        Acsr[tmp2] = Acoo[i];
        ia[tmp1]++;
    }

    for (i = 0, tmp2 = 0; i <= *n; i++) {
        tmp1 = ia[i];
        ia[i] = tmp2;
        tmp2 = tmp1;
    }

    mat->row = ia;
    mat->col = ja;
    mat->value = Acsr;
    mat->format = "csr";

    return 0;
}

