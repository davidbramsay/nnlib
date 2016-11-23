#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>



matrix_t *m_init(int num_rows, int num_cols){
//initialize a matrix object, setting all values to zero

    matrix_t *matrix = malloc(sizeof(matrix_t));
    matrix->num_rows = num_rows;
    matrix->num_cols = num_cols;

    matrix->vals = (double **)malloc(sizeof(double *) * num_rows);

    for (int i = 0; i < num_rows; i++){

        //initialize each row in memory
        matrix->vals[i] = (double *)malloc(sizeof(double) * num_cols);
        memset(matrix->vals[i], 0, sizeof(double) * num_cols);

    }

    return matrix;
}



matrix_t *m_init_array(double *array[], int num_rows, int num_cols){
//initialize matrix object with array values

    matrix_t *matrix = malloc(sizeof(matrix_t));
    matrix->num_rows = num_rows;
    matrix->num_cols = num_cols;

    matrix->vals = (double **)malloc(sizeof(double *) * num_rows);

    for (int i = 0; i < num_rows; i++){

        //initialize each row in memory
        matrix->vals[i] = (double *)malloc(sizeof(double) * num_cols);

        //copy values from passed matrix to our copy
        memcpy((void *)(matrix->vals[i]), (void *)(array+i*num_cols),
                sizeof(double) * matrix->num_cols);
    }

    return matrix;
}



matrix_t *m_copy(matrix_t *matrix){
//deep copy of a matrix object

    matrix_t *matrix_copy = malloc(sizeof(matrix_t));
    matrix_copy->num_rows = matrix->num_rows;
    matrix_copy->num_cols = matrix->num_cols;

    matrix_copy->vals = (double **)malloc(sizeof(double *) * matrix->num_rows);

    for (int i = 0; i < matrix->num_rows; i++){

        //allocate each row in memory
        matrix_copy->vals[i] = (double *)malloc(sizeof(double) * matrix->num_cols);

        //copy values from passed matrix to our copy
        memcpy((void *)(matrix_copy->vals[i]), (void *)(matrix->vals[i]),
                sizeof(double) * matrix->num_cols);

    }

    return matrix_copy;
}



void m_copy_into(matrix_t *src, matrix_t *dest){
//deep copy of a matrix object from src into dest

    if (src->num_rows > dest->num_rows || src->num_cols > dest->num_cols){
        fprintf(stderr, "m_copy_into: FAILED: DEST MATRIX HAS NOT ALLOCATED ENOUGH MEMORY TO HOLD SRC\n");
        return;
    }

    dest->num_rows = src->num_rows;
    dest->num_cols = src->num_cols;

    for (int i = 0; i < src->num_rows; i++){

        //copy values from passed matrix to our copy
        memcpy((void *)(dest->vals[i]), (void *)(src->vals[i]),
                sizeof(double) * src->num_cols);

    }

}



void m_free(matrix_t *matrix){
//step through malloced values and free them, for a matrix

    for (int i = 0; i < matrix->num_rows; i++){
        free(matrix->vals[i]);
    }

    free(matrix->vals);
    free(matrix);

}



void m_randomize_vals(matrix_t *matrix, int upper, int lower){
//fill matrix will random values between lower and upper

    srand(time(NULL));

    for (int i = 0; i < matrix->num_rows; i++)
        for (int j = 0; j < matrix->num_cols; j++)
            matrix->vals[i][j] = ((double)(upper-lower) * (double)rand() / RAND_MAX) + lower;
}



matrix_t *m_transpose(matrix_t *matrix){
//transpose a matrix
    matrix_t *result_matrix = m_init(matrix->num_cols, matrix->num_rows);
    for (int i = 0; i < matrix->num_rows; i++)
        for (int j = 0; j < matrix->num_cols; j++)
            result_matrix->vals[j][i] = matrix->vals[i][j];

    return result_matrix;
}



matrix_t *m_dist(matrix_t *matrix1, matrix_t *matrix2){
//calc euclidean distance between rows

    if (matrix1->num_rows != matrix2->num_rows){
        fprintf(stderr, "m_dist: ERROR- ROWS DON'T MATCH: %dx%d with %dx%d, rows must match\n",
                matrix1->num_rows, matrix1->num_cols, matrix2->num_rows, matrix2->num_cols);
        return NULL;
    }
    if (matrix1->num_cols != matrix2->num_cols){
        fprintf(stderr, "m_dist: WARNING - COLS DON'T MATCH: %dx%d with %dx%d, will proceed.\n",
                matrix1->num_rows, matrix1->num_cols, matrix2->num_rows, matrix2->num_cols);
    }

    int maxlen = MAX(matrix1->num_cols, matrix2->num_cols);
    matrix_t *result = m_init(matrix1->num_rows, 1);

    for (int i = 0; i < maxlen; i++){
        if (i >= matrix1->num_cols)
            for (int j = 0; j < result->num_rows; j++)
                result->vals[j][0] += (0 - matrix2->vals[j][i]) * (0 - matrix2->vals[j][i]);
        else if (i >= matrix2->num_cols)
            for (int j = 0; j < result->num_rows; j++)
                result->vals[j][0] += (0 - matrix1->vals[j][i]) * (0 - matrix1->vals[j][i]);
        else
            for (int j = 0; j < result->num_rows; j++)
                result->vals[j][0] += (matrix1->vals[j][i] - matrix2->vals[j][i]) *
                    (matrix1->vals[j][i] - matrix2->vals[j][i]);
    }

    for (int i = 0; i < result->num_rows; i++)
        result->vals[i][0] = sqrt(result->vals[i][0]);

    return result;
}



matrix_t *m_maxval(matrix_t *matrix){
//find largest abs in each row

    matrix_t *result = m_init(matrix->num_rows, 1);

    for (int i = 0; i < matrix->num_rows; i++){

        double maxval = 0;

        for (int j = 0; j < matrix->num_cols; j++)
            if (fabs(matrix->vals[i][j]) > maxval)
                maxval = fabs(matrix->vals[i][j]);

        result->vals[i][0] = maxval;

    }

    return result;
}



matrix_t *m_maxind(matrix_t *matrix){
//index of largest abs in each row

    matrix_t *result = m_init(matrix->num_rows, 1);

    for (int i = 0; i < matrix->num_rows; i++){

        double maxval = 0;
        int index = 0;

        for (int j = 0; j < matrix->num_cols; j++)
            if (fabs(matrix->vals[i][j]) > maxval){
                maxval = fabs(matrix->vals[i][j]);
                index = j;
            }

        result->vals[i][0] = (double)index;

    }

    return result;

}



double m_dmaxval(matrix_t *matrix){
//find largest abs in all of matrix, return val

    double maxval = 0;

    for (int i = 0; i < matrix->num_rows; i++)
        for (int j = 0; j < matrix->num_cols; j++)
            if (fabs(matrix->vals[i][j]) > fabs(maxval))
                maxval = matrix->vals[i][j];

    return maxval;
}



matrix_t *m_getrow(matrix_t *matrix, int row) {
//return matrix of row [1][num_cols]
    if (row >= matrix->num_rows) {
        fprintf(stderr, "m_getrow: row out of bounds, matrix num rows is %d, tried to access  %d\n",
                matrix->num_rows, row);
        return NULL;
    }

    matrix_t *result = m_init(1, matrix->num_cols);

    for (int i = 0; i < matrix->num_cols; i++)
        result->vals[0][i] = matrix->vals[row][i];

    return result;
}



matrix_t *m_getcol(matrix_t *matrix, int col) {
//return vector of col [num_rows][1]

    if (col >= matrix->num_cols) {
        fprintf(stderr, "m_getcol: col out of bounds, matrix num cols is %d, tried to access  %d\n",
                matrix->num_cols, col);
        return NULL;
    }

    matrix_t *result = m_init(matrix->num_rows, 1);

    for (int i = 0; i < matrix->num_rows; i++)
        result->vals[i][0] = matrix->vals[i][col];

    return result;
}



matrix_t *m_multi(matrix_t *matrix1, matrix_t *matrix2){
//multiplication of two matricies, return null if dims don't match

    //check if inner dimension is the same, error if not
    if (matrix1->num_cols != matrix2->num_rows){

        fprintf(stderr, "m_multi: DIMS DON'T MATCH: %dx%d with %dx%d, (1) cols must match (2) rows\n",
                matrix1->num_rows, matrix1->num_cols, matrix2->num_rows, matrix2->num_cols);

        return NULL;
    }

    //do the multiplication
    matrix_t *result_matrix = m_init(matrix1->num_rows, matrix2->num_cols);

    for (int i = 0; i < result_matrix->num_rows; i++)
        for (int j = 0; j < result_matrix->num_cols; j++)
            for (int k = 0; k < matrix1->num_cols; k++)
                result_matrix->vals[i][j] += matrix1->vals[i][k]*matrix2->vals[k][j];

    return result_matrix;
}



void m_pmulti(matrix_t *matrix1, matrix_t *matrix2){
//pointwise multiplication of two matricies, update matrix1

    //check if dimensions are the same, error if not
    if (matrix1->num_rows != matrix2->num_rows ||
            matrix1->num_cols != matrix2->num_cols){

        fprintf(stderr, "m_pmulti: DIMS DON'T MATCH: %dx%d pointwise with %dx%d\n",
                matrix1->num_rows, matrix1->num_cols, matrix2->num_rows, matrix2->num_cols);

    }

    //do the multiplication
    for (int i = 0; i < matrix1->num_rows; i++)
        for (int j = 0; j < matrix1->num_cols; j++)
            matrix1->vals[i][j] *= matrix2->vals[i][j];

}



void m_smulti(matrix_t *matrix, double scalar){
//scalar multiplication for all values of a matrix

    for (int i = 0; i < matrix->num_rows; i++)
        for (int j = 0; j < matrix->num_cols; j++)
                matrix->vals[i][j] *= scalar;

}



void m_padd(matrix_t *matrix1, matrix_t *matrix2){
//pointwise addition of two matricies, update matrix1

    //check if dimensions are the same, error if not
    if (matrix1->num_rows != matrix2->num_rows ||
            matrix1->num_cols != matrix2->num_cols){

        fprintf(stderr, "m_padd: DIMS DON'T MATCH: %dx%d pointwise with %dx%d\n",
                matrix1->num_rows, matrix1->num_cols, matrix2->num_rows, matrix2->num_cols);

    }

    //do the addition
    for (int i = 0; i < matrix1->num_rows; i++)
        for (int j = 0; j < matrix1->num_cols; j++)
            matrix1->vals[i][j] += matrix2->vals[i][j];

}



void m_psub(matrix_t *matrix1, matrix_t *matrix2){
//pointwise subtraction of two matricies, update matrix1

    //check if dimensions are the same, error if not
    if (matrix1->num_rows != matrix2->num_rows ||
            matrix1->num_cols != matrix2->num_cols){

        fprintf(stderr, "m_psub: DIMS DON'T MATCH: %dx%d pointwise with %dx%d\n",
                matrix1->num_rows, matrix1->num_cols, matrix2->num_rows, matrix2->num_cols);

    }

    //do the subtraction
    for (int i = 0; i < matrix1->num_rows; i++)
        for (int j = 0; j < matrix1->num_cols; j++)
            matrix1->vals[i][j] -= matrix2->vals[i][j];

}



void m_sadd(matrix_t *matrix, double scalar){
//scalar addition for all values of a matrix

    for (int i = 0; i < matrix->num_rows; i++)
        for (int j = 0; j < matrix->num_cols; j++)
            matrix->vals[i][j] += scalar;

}



void m_normalize(matrix_t *matrix){
//normalize by num_cols

    for (int i = 0; i < matrix->num_rows; i++)
        for (int j = 0; j < matrix->num_cols; j++)
            matrix->vals[i][j] /= matrix->num_cols;
}



void m_normalize_nonzero(matrix_t *matrix){
//normalize each row by # nonzero elements in that row

    for (int i = 0; i < matrix->num_rows; i++){

        int num_nonzero = 0;

        for (int j = 0; j < matrix->num_cols; j++)
            if (matrix->vals[i][j]) num_nonzero++;

        for (int j = 0; j < matrix->num_cols; j++)
            matrix->vals[i][j] /= num_nonzero;
    }
}



void m_apply(matrix_t *matrix, double (*func)(double)){
//apply function pointwise to matrix

    for (int i = 0; i < matrix->num_rows; i++)
        for (int j = 0; j < matrix->num_cols; j++)
            matrix->vals[i][j] = func(matrix->vals[i][j]);

}



void m_print(matrix_t *matrix){
//print a matrix with tabs in between each value

    for (int i = 0; i < matrix->num_rows; i++){

        for (int j = 0; j < matrix->num_cols; j++)
            printf("%.2f\t", matrix->vals[i][j]);

        printf("\n");
    }
}



void m_upper_triangular(matrix_t *matrix) {
//upper triangular, gaussian elimination

    //copy matrix in case elimination fails
    matrix_t *temp = m_copy(matrix);

    //proceed with elimination, main elimination loop
    for (int i = 0; i < matrix->num_rows; i++){

        if (matrix->vals[i][i] == 0){ //check if pivot is zero

            //if it is, find a row to switch with and switch
            for (int m = i + 1; m < matrix->num_rows; m++){
                if (matrix->vals[m][i] != 0){

                    //switch row i and m, and move on (not worrying  about
                    //cols before pivot, as they are all zero)
                    for (int n = i; n < matrix->num_cols; n++){

                        double temp = matrix->vals[i][n];
                        matrix->vals[i][n] = matrix->vals[m][n];
                        matrix->vals[m][n] = temp;
                    }

                    continue;
                }
            }
            //check if we successfully switched, or if we're singular
            if (matrix->vals[i][i] == 0){

                //we're singular
                fprintf(stderr, "m_upper_triangular: ERROR SINGULAR MATRIX:\n");
                m_print(matrix);

                //restore previous matrix, free our triangular, throw error, return
                m_copy_into(temp, matrix);
                m_free(temp);
                return;
            }
        }

        //actually do the elimination
        for (int m = i + 1; m < matrix->num_rows; m++){

            //subtract (val[m][i]/val[i][i])*val[i][...] from val[m][...]
            double elim_const = matrix->vals[m][i] / matrix->vals[i][i];

            for (int n = i; n < matrix->num_cols; n++){
                matrix->vals[m][n] -= elim_const * matrix->vals[i][n];
            }
        }
    }

    m_free(temp);

}



double cube(double in){
    return in*in*in;
}



void unit_test(){

    //basic use tests

    //basic test #1
    matrix_t *x = m_init(5, 10);

    printf("init 5x10 matrix\n");
    printf("rows = %d\n", x->num_rows);
    printf("cols = %d\n", x->num_cols);

    printf("(0) x[3][3] = %f\n", x->vals[3][3]);
    printf("(0) x[4][9] = %f\n", x->vals[4][9]);

    x->vals[4][9] = 5.5;
    printf("(5.5) x[4][9] = %f\n", x->vals[4][9]);

    m_print(x);

    matrix_t *x2 = m_transpose(x);
    printf("(5.5) x2[9][4] = %f\n", x2->vals[9][4]);

    m_print(x2);

    m_free(x);
    m_free(x2);

    //basic test #2
    /*
    x = m_init(3,2);
    x->vals[0][0]=4;
    x->vals[0][1]=2;
    x->vals[1][0]=1;
    x->vals[1][1]=-1;
    x->vals[2][0]=3;
    x->vals[2][1]=5;
    */

    double array[3][2] = {{4,2},{1,-1},{3,5}};
    x = m_init_array((double **)array,3,2);
    printf("x = 3x2\n");
    m_print(x);

    x2 = m_transpose(x);
    printf("x2 = 2x3, transposed x\n");
    m_print(x2);

    printf("x3 = mult x with x2\n");
    matrix_t *x3 = m_multi(x, x2);
    m_free(x2);
    m_print(x3);
    m_free(x3);

    printf("x = pointwise multi x with x\n");
    matrix_t *x4 = m_copy(x);
    m_pmulti(x, x4);
    m_print(x);
    m_free(x4);

    printf("x = scale x by 1/2\n");
    matrix_t *x5 = m_copy(x);
    m_free(x);
    m_smulti(x5, 0.5);
    m_print(x5);

    printf("x = add x to x\n");
    matrix_t *x6 = m_copy(x5);
    m_free(x5);
    m_padd(x6, x6);
    m_print(x6);

    printf("x = add 5 to x\n");
    m_sadd(x6, 5);
    m_print(x6);

    printf("x = x cubed, by passing function cube to m_apply\n");
    m_apply(x6, &cube);
    m_print(x6);
    m_free(x6);


    //linear algebra section
    double array2[3][3] = {{1,1,1},{2,2,5},{4,6,8}};
    x = m_init_array((double **)array2,3,3);
    printf("x = 3x3\n");
    m_print(x);
    printf("x = upper gaussian eliminated\n");
    m_upper_triangular(x);
    m_print(x);
    free(x);

    double array3[3][3] = {{1,2,4},{1,2,4},{1,5,8}};
    x = m_init_array((double **)array3,3,3);
    printf("x = 3x3\n");
    m_print(x);
    printf("x = upper gaussian eliminated, should fail as singular\n");
    m_upper_triangular(x);
    printf("upper failed, restored to previous:\n");
    m_print(x);
    free(x);

    double array4[3][4] = {{1,3,1,9},{1,1,-1,1},{3,11,5,35}};
    x = m_init_array((double **)array4,3,4);
    printf("x = 4x5\n");
    m_print(x);
    printf("x = upper gaussian eliminated\n");
    m_upper_triangular(x);
    m_print(x);
    m_free(x);

    x = m_init(4,3);
    //m_randomize_weights(x);
    printf("randomized 4x3\n");
    m_print(x);
    printf("max vals\n");
    x2 = m_maxval(x);
    m_print(x2);
    printf("max inds\n");
    x2 = m_maxind(x);
    m_print(x2);
    printf("normalized\n");
    m_normalize(x);
    m_print(x);
    m_free(x);
    m_free(x2);

}

