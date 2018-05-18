/*
 * @author: João Gabriel Basi Nº USP: 9793801
 * @author: Juliano Garcia de Oliveira Nº USP: 9277086
 *
 * MAC0219
 * 26/03/19
 *
 * A header with some macros
 */

#ifndef __MACROS_H__
#define __MACROS_H__

#include <pthread.h>
#include <math.h>
#include <inttypes.h>

#define P(x) pthread_mutex_lock(x)
#define V(x) pthread_mutex_unlock(x)

/* Simple types definition */
typedef enum { false, true } bool;

typedef unsigned int u_int;

typedef unsigned long long int u_lint;

typedef struct mat_t{
    double **matrix;
    uint64_t n;
    uint64_t m;
} mat;

typedef struct mat_c_t{
    double *m_c;
    uint64_t n;
    uint64_t m;
    uint64_t nm;
} mat_c;

typedef struct targ_t {
    double **A, **B, **C;
    uint64_t ini_ar, ini_ac, ini_br, ini_bc, ini_cr, ini_cc;
    uint64_t size_ar, size_ac, size_bc, min_size;
} targ;

typedef mat* Matrix;

typedef mat_c* MatrixArray;

typedef targ* Argument;



/* Functions */
MatrixArray new_matrixArray(char* filename);
MatrixArray new_matrixArray_clean(uint64_t n, uint64_t m);
void destroy_matrixArray(MatrixArray mtxArr);
void write_matrixArray(MatrixArray mtxArr, char* filename);
void reset_matrixArray(MatrixArray mtxArr);
void print_matrixArray(MatrixArray mtxArr);

/*
 * Function: new_matrix
 * --------------------------------------------------------
 * Creates a new matrix given the filename of the matrix
 *
 * @args  filename :  string
 *
 * @return  the new matrix created
 */
Matrix new_matrix(char* filename);

/*
 * Function: new_matrix_clean
 * --------------------------------------------------------
 * Creates a new matrix with dim (n x m) filled with zeros
 *
 * @args  n : number of rows
 *        m : number of cols
 *
 * @return  the new matrix created
 */
Matrix new_matrix_clean(uint64_t n, uint64_t m);

/*
 * Function: destroy_matrix
 * --------------------------------------------------------
 * Free the specified matrix
 *
 * @args  mtx :  the Matrix
 *
 * @return
 */
void destroy_matrix(Matrix mtx);

/*
 * Function: write_matrix
 * --------------------------------------------------------
 * Writes out the matrix in the format specified in a file
 *
 * @args  mtx :  the Matrix
 *        filename: the name of the file
 *
 * @return
 */
void write_matrix(Matrix mtx, char* filename);

/*
 * Function: reset_matrix
 * --------------------------------------------------------
 * Set all the entries in the matrix to zero
 *
 * @args  mtx :  the Matrix
 *
 * @return
 */
void reset_matrix(Matrix mtx);

Argument create_argument(double** A, uint64_t ini_ar, uint64_t ini_ac,
                         double** B, uint64_t ini_br, uint64_t ini_bc,
                         double** C, uint64_t ini_cr, uint64_t ini_cc,
                         uint64_t size_ar, uint64_t size_ac, uint64_t size_bc, uint64_t min_size);

/* TODO: REMOVE THIS, DEBUGGER */
void print_matrix(Matrix mtx);

bool are_equal_ma2m(MatrixArray ma, Matrix m);

#endif
