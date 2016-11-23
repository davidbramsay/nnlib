#ifndef MATRIX_H
#define MATRIX_H

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

//to access 2D array values, matrix_t matrix.vals[x][y]
typedef struct matrix{

    double **vals;
    int num_rows;
    int num_cols;

} matrix_t;


//initialize matrix and memory management
matrix_t *m_init(int num_rows, int num_cols);
matrix_t *m_init_array(double *array[], int num_rows, int num_cols);
matrix_t *m_copy(matrix_t *matrix);

void m_copy_into(matrix_t *src, matrix_t *dest);
void m_free(matrix_t *matrix);

void m_randomize_vals(matrix_t *matrix, int upper, int lower);

//operations that result in new matricies or values
matrix_t *m_transpose(matrix_t *matrix); //transpose matrix
matrix_t *m_dist(matrix_t *matrix1, matrix_t *matrix2); //calc euclidean distance between rows

matrix_t *m_maxval(matrix_t *matrix); //find largest abs in each row
matrix_t *m_maxind(matrix_t *matrix); //index of largest abs in each row

double m_dmaxval(matrix_t *matrix); //find largest abs in all of matrix

matrix_t *m_getrow(matrix_t *matrix, int row); //return matrix of row [1][num_cols]
matrix_t *m_getcol(matrix_t *matrix, int col); //return vector of col [num_rows][1]

matrix_t *m_multi(matrix_t *matrix1, matrix_t *matrix2); //standard matrix multiplication

//in place operations on first matrix argument
void m_pmulti(matrix_t *matrix1, matrix_t *matrix2); //pointwise multiplication, matrix1 updated
void m_smulti(matrix_t *matrix, double scalar); //scalar multiplication

void m_padd(matrix_t *matrix1, matrix_t *matrix2); //pointwise addition, matrix1 updated
void m_sadd(matrix_t *matrix, double scalar); //scalar addition

void m_psub(matrix_t *matrix1, matrix_t *matrix2); //pointwise subtraction, matrix1 updated

void m_normalize(matrix_t *matrix); //normalize by num_cols
void m_normalize_nonzero(matrix_t *matrix); //normalize each row by # nonzero elements in that row


void m_apply(matrix_t *matrix, double (*func)(double)); //apply function pointwise to matrix



//helper functions
void m_print(matrix_t *matrix); //print matrix to screen

//linear algebra
void m_upper_triangular(matrix_t *matrix); //upper triangular, gaussian elimination


#endif /* MATRIX_H */
