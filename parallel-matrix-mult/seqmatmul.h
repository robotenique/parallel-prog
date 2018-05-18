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

void add(double** C, int ini_cr, int ini_cc,
         double** T, int ini_tr, int ini_tc,
         int size_r, int size_c);

void matmul_seq(Matrix A, Matrix B, Matrix C, int min_size);

//TODO: REMOVE THIS
void matmul_trashy(Matrix A, Matrix B, Matrix C);
void matcoisa(MatrixArray A, MatrixArray B, MatrixArray C);

#endif
