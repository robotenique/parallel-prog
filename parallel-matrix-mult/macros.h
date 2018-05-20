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

#define MIN_SIZE 256

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
    double *A, *B, *C;
    uint64_t ini_ar, ini_ac, ini_br, ini_bc, ini_cr, ini_cc;
    uint64_t or_size_ac, or_size_bc;
    uint64_t size_ar, size_ac, size_bc, min_size, num_threads;
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

/*
 * Function: create_argument
 * --------------------------------------------------------
 * Create an Argument to be passed to matmul_pt
 *
 * @args  A : pointer to the first position of the first MatrixArray
 *        B : pointer to the first position of the second MatrixArray
 *        C : pointer to the first position of the result MatrixArray
 *        size_ar : number of rows of A
 *        size_ac : number of collumns of A
 *        size_bc : number of collumns of B
 *        or_size_ac : original number of collumns of A
 *        or_size_ar : original number of collumns of B
 *        min_size : minimum matrix size for the trivial algorithm to be executed
 *        num_threads : number of threads opened
 *
 * @return an Argument struct with all the informations
 */
Argument create_argument(double* A, double* B, double* C,
                         uint64_t size_ar, uint64_t size_ac, uint64_t size_bc,
                         uint64_t or_size_ac, uint64_t or_size_bc,
                         uint64_t num_threads);

/* TODO: REMOVE THIS, DEBUGGER */
void print_matrix(Matrix mtx);

void print_matrixArray(MatrixArray mtxArr);

bool are_equal_ma2m(MatrixArray ma, Matrix m);

bool are_equal_ma2ma(MatrixArray ma, MatrixArray m);

void print_num_threads();

#endif
