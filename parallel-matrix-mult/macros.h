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

#define P(x) pthread_mutex_lock(x)
#define V(x) pthread_mutex_unlock(x)

/* Simple types definition */
typedef enum { false, true } bool;

typedef unsigned int u_int;

typedef unsigned long long int u_lint;

typedef struct mat_t{
    double **matrix;
    u_int n;
    u_int m;
} mat;

typedef mat* Matrix;



/* Functions */

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
Matrix new_matrix_clean(u_int n, u_int m);

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

/* TODO: REMOVE THIS, DEBUGGER */
void print_matrix(Matrix mtx);
#endif