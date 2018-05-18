#include <stdio.h>
#include <stdlib.h>
#include "macros.h"

void matmul_seq_rec(double** A, int ini_ar, int ini_ac,
                    double** B, int ini_br, int ini_bc,
                    double** C, int ini_cr, int ini_cc,
                    int size_ar, int size_ac, int size_bc, int min_size) {
    if (!size_ac || !size_ar || !size_bc)
        return ;
    if (size_ar <= min_size && size_ac <= min_size && size_bc <= min_size) {
        for (int i = 0; i < size_ar; i++) {
            for (int k = 0; k < size_ac; k++) {
                for (int j = 0; j < size_bc; j++)
                    C[ini_cr+i][ini_cc+j] += A[ini_ar+i][ini_ac+k]*B[ini_br+k][ini_bc+j];
            }
        }
        return ;
    }
    int new_size_ar = size_ar/2;
    int new_size_ac = size_ac/2;
    int new_size_bc = size_bc/2;

    matmul_seq_rec(A, ini_ar, ini_ac,
                   B, ini_br, ini_bc,
                   C, ini_cr, ini_cc,
                   new_size_ar, new_size_ac, new_size_bc, min_size);
    matmul_seq_rec(A, ini_ar, ini_ac,
                   B, ini_br, ini_bc + new_size_bc,
                   C, ini_cr, ini_cc + new_size_bc,
                   new_size_ar, new_size_ac, size_bc - new_size_bc, min_size);
    matmul_seq_rec(A, ini_ar + new_size_ar, ini_ac,
                   B, ini_br, ini_bc,
                   C, ini_cr + new_size_ar, ini_cc,
                   size_ar - new_size_ar, new_size_ac, new_size_bc, min_size);
    matmul_seq_rec(A, ini_ar + new_size_ar, ini_ac,
                   B, ini_br, ini_bc + new_size_bc,
                   C, ini_cr + new_size_ar, ini_cc + new_size_bc,
                   size_ar - new_size_ar, new_size_ac, size_bc - new_size_bc, min_size);

    matmul_seq_rec(A, ini_ar, ini_ac + new_size_ac,
                   B, ini_br + new_size_ac, ini_bc,
                   C, ini_cr, ini_cc,
                   new_size_ar, size_ac - new_size_ac, new_size_bc, min_size);
    matmul_seq_rec(A, ini_ar, ini_ac + new_size_ac,
                   B, ini_br + new_size_ac, ini_bc + new_size_bc,
                   C, ini_cr, ini_cc + new_size_bc,
                   new_size_ar, size_ac - new_size_ac, size_bc - new_size_bc, min_size);
    matmul_seq_rec(A, ini_ar + new_size_ar, ini_ac + new_size_ac,
                   B, ini_br + new_size_ac, ini_bc,
                   C, ini_cr + new_size_ar, ini_cc,
                   size_ar - new_size_ar, size_ac - new_size_ac, new_size_bc, min_size);
    matmul_seq_rec(A, ini_ar + new_size_ar, ini_ac + new_size_ac,
                   B, ini_br + new_size_ac, ini_bc + new_size_bc,
                   C, ini_cr + new_size_ar, ini_cc + new_size_bc,
                   size_ar - new_size_ar, size_ac - new_size_ac, size_bc - new_size_bc, min_size);

    return ;
}

void matmul_seq(Matrix A, Matrix B, Matrix C, int min_size) {
    matmul_seq_rec(A->matrix, 0, 0, B->matrix, 0, 0, C->matrix, 0, 0, A->n, A->m, B->m, min_size);
}

void matmul_trashy(Matrix A, Matrix B, Matrix C){
    double **a = A->matrix;
    double **b = B->matrix;
    double **c = C->matrix;
    for (size_t i = 0; i < A->n; i++)
        for (size_t j = 0; j < B->m; j++)
            for (size_t k = 0; k < A->m; k++)
                c[i][j] += a[i][k]*b[k][j];
}
void matcoisa(MatrixArray A, MatrixArray B, MatrixArray C){
    u_int a_n = A->n;
    u_int a_m = A->m;
    u_int b_n = B->n;
    u_int b_m = B->m;
    u_int c_n = C->n;
    u_int c_m = C->m;
    double* a = A->m_c;
    double* b = B->m_c;
    double* c = C->m_c;

    int i, j, k;
    for (i = 0; i < a_n; i++) {
         //printf("Thread #%d is doing row %d.\n",th_id,i);
         for (k = 0; k < b_n; k++) {
            double r = *a++;
            double *pB = b + k*b_m;
            double *pC = c + i*b_m;
            for (j = 0; j < b_m; j++) {
                *pC++ += r* *pB++;
                //c[i*c_m + j] += r*b[k*b_m + j];
            }
         }
    }
}
