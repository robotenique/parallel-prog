/*
 * @author: João Gabriel Basi Nº USP: 9793801
 * @author: Juliano Garcia de Oliveira Nº USP: 9277086
 *
 * MAC0219
 * 26/03/19
 *
 * Some functions for sequential matrix multiplication
 */

#ifndef __SEQMATMUL_H__
#define __SEQMATMUL_H__

#define MIN_SIZE 256
#include <inttypes.h>
void add(double** C, uint64_t ini_cr, uint64_t ini_cc,
         double** T, uint64_t ini_tr, uint64_t ini_tc,
         uint64_t size_r, uint64_t size_c);

void matmul_seq(Matrix A, Matrix B, Matrix C, uint64_t min_size);

//TODO: REMOVE THIS
void matmul_trashy(Matrix A, Matrix B, Matrix C);
void matcoisa(MatrixArray A, MatrixArray B, MatrixArray C);

#endif
