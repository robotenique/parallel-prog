/*
 * @author: João Gabriel Basi Nº USP: 9793801
 * @author: Juliano Garcia de Oliveira Nº USP: 9277086
 *
 * MAC0219
 * 26/03/19
 *
 * Some functions for parallel matrix multiplication using pthreads
 */

#ifndef __PTMATMUL_H__
#define __PTMATMUL_H__

#include "macros.h"
#include <inttypes.h>

#define MAX_THREADS 5000

/*
 * Function: matmul_pt
 * --------------------------------------------------------
 * This function does a matrix multiplication of AB and put the result
 * in C, by using pthread.
 *
 * @args A : MatrixArray
 *       B : MatrixArray
 *       C : MatrixArray
 *       min_size : minimum matrix size for the trivial algorithm to be executed
 *
 * @return
 */

void matmul_pt(MatrixArray A, MatrixArray B, MatrixArray C);

#endif
