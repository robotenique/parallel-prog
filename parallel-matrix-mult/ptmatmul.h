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

void matmul_pt(MatrixArray A, MatrixArray B, MatrixArray C, uint64_t min_size);

#endif
