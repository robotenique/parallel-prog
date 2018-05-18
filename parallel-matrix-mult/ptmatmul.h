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

void matmul_pt(Matrix A, Matrix B, Matrix C, u_int min_size);

#endif
