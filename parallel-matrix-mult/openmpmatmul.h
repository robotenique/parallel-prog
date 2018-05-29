/*
 * @author: João Gabriel Basi Nº USP: 9793801
 * @author: Juliano Garcia de Oliveira Nº USP: 9277086
 *
 * MAC0219
 *
 * A header for the matrix multiplication using openMP
 */

#ifndef __OPENMPMATMUL_H__
#define __OPENMPMATMUL_H__

/*
 * Function: matmul_omp
 * --------------------------------------------------------
 * This function does a matrix multiplication of AB and put the result
 * in C, by using openMP.
 *
 * @args A : MatrixArray
 *       B : MatrixArray
 *       C : MatrixArray
 *
 * @return
 */
void matmul_omp(MatrixArray A, MatrixArray B, MatrixArray C);

void matmul_omp2(MatrixArray A, MatrixArray B, MatrixArray C);

#endif
