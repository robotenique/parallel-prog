/*
 * @author: João Gabriel Basi Nº USP: 9793801
 * @author: Juliano Garcia de Oliveira Nº USP: 9277086
 *
 * MAC0219
 *
 * Some functions for sequential matrix multiplication
 */
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
