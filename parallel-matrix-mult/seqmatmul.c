#include <stdio.h>
#include <inttypes.h>
#include <stdlib.h>
#include "macros.h"

void matmul_seq_rec(double** A, uint64_t ini_ar, uint64_t ini_ac,
                    double** B, uint64_t ini_br, uint64_t ini_bc,
                    double** C, uint64_t ini_cr, uint64_t ini_cc,
                    uint64_t size_ar, uint64_t size_ac, uint64_t size_bc) {
    if (!size_ac || !size_ar || !size_bc)
        return ;
    if (size_ar <= MIN_SIZE && size_ac <= MIN_SIZE && size_bc <= MIN_SIZE) {
        for (uint64_t i = 0; i < size_ar; i++) {
            for (uint64_t k = 0; k < size_ac; k++) {
                for (uint64_t j = 0; j < size_bc; j++)
                    C[ini_cr+i][ini_cc+j] += A[ini_ar+i][ini_ac+k]*B[ini_br+k][ini_bc+j];
            }
        }
        return ;
    }
    uint64_t new_size_ar = size_ar/2;
    uint64_t new_size_ac = size_ac/2;
    uint64_t new_size_bc = size_bc/2;

    matmul_seq_rec(A, ini_ar, ini_ac,
                   B, ini_br, ini_bc,
                   C, ini_cr, ini_cc,
                   new_size_ar, new_size_ac, new_size_bc);
    matmul_seq_rec(A, ini_ar, ini_ac,
                   B, ini_br, ini_bc + new_size_bc,
                   C, ini_cr, ini_cc + new_size_bc,
                   new_size_ar, new_size_ac, size_bc - new_size_bc);
    matmul_seq_rec(A, ini_ar + new_size_ar, ini_ac,
                   B, ini_br, ini_bc,
                   C, ini_cr + new_size_ar, ini_cc,
                   size_ar - new_size_ar, new_size_ac, new_size_bc);
    matmul_seq_rec(A, ini_ar + new_size_ar, ini_ac,
                   B, ini_br, ini_bc + new_size_bc,
                   C, ini_cr + new_size_ar, ini_cc + new_size_bc,
                   size_ar - new_size_ar, new_size_ac, size_bc - new_size_bc);

    matmul_seq_rec(A, ini_ar, ini_ac + new_size_ac,
                   B, ini_br + new_size_ac, ini_bc,
                   C, ini_cr, ini_cc,
                   new_size_ar, size_ac - new_size_ac, new_size_bc);
    matmul_seq_rec(A, ini_ar, ini_ac + new_size_ac,
                   B, ini_br + new_size_ac, ini_bc + new_size_bc,
                   C, ini_cr, ini_cc + new_size_bc,
                   new_size_ar, size_ac - new_size_ac, size_bc - new_size_bc);
    matmul_seq_rec(A, ini_ar + new_size_ar, ini_ac + new_size_ac,
                   B, ini_br + new_size_ac, ini_bc,
                   C, ini_cr + new_size_ar, ini_cc,
                   size_ar - new_size_ar, size_ac - new_size_ac, new_size_bc);
    matmul_seq_rec(A, ini_ar + new_size_ar, ini_ac + new_size_ac,
                   B, ini_br + new_size_ac, ini_bc + new_size_bc,
                   C, ini_cr + new_size_ar, ini_cc + new_size_bc,
                   size_ar - new_size_ar, size_ac - new_size_ac, size_bc - new_size_bc);

    return ;
}

void matmul_seq(Matrix A, Matrix B, Matrix C) {
    matmul_seq_rec(A->matrix, 0, 0, B->matrix, 0, 0, C->matrix, 0, 0, A->n, A->m, B->m);
}

void matmul_trashy(Matrix A, Matrix B, Matrix C){
    double **a = A->matrix;
    double **b = B->matrix;
    double **c = C->matrix;
    for (uint64_t i = 0; i < A->n; i++)
        for (uint64_t j = 0; j < B->m; j++)
            for (uint64_t k = 0; k < A->m; k++)
                c[i][j] += a[i][k]*b[k][j];
}
void matcoisa(MatrixArray A, MatrixArray B, MatrixArray C){
    uint64_t a_n = A->n;
    uint64_t a_m = A->m;
    uint64_t b_n = B->n;
    uint64_t b_m = B->m;
    uint64_t c_n = C->n;
    uint64_t c_m = C->m;
    double* a = A->m_c;
    double* b = B->m_c;
    double* c = C->m_c;

    uint64_t i, j, k;
    for (i = 0; i < a_n; i++) {
         //printf("Thread #%lu is doing row %lu.\n",th_id,i);
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
