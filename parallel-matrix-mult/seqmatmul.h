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

#define MIN_SIZE 512

void matmul_seq(Matrix A, Matrix B, Matrix C, int min_size);
void matmul_seq2(Matrix A, Matrix B, Matrix C, int min_size);

#endif
