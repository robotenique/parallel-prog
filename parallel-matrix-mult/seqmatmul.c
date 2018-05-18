#include <stdio.h>
#include <stdlib.h>
#include "error.h"
#include "macros.h"

void add(double** C, int ini_cr, int ini_cc,
         double** T, int ini_tr, int ini_tc,
         int size_r, int size_c) {
    if (!size_r || !size_c)
        return ;
    if (size_r == 1 && size_c == 1) {
        C[ini_cr][ini_cc] += T[ini_tr][ini_tc];
        return ;
    }
    int new_size_r = size_r/2;
    int new_size_c = size_c/2;
    add(C, ini_cr, ini_cc,
        T, ini_tr, ini_tc,
        new_size_r, new_size_c);
    add(C, ini_cr + new_size_r, ini_cc,
        T, ini_tr + new_size_r, ini_tc,
        size_r - new_size_r, new_size_c);
    add(C, ini_cr, ini_cc + new_size_c,
        T, ini_tr, ini_tc + new_size_c,
        new_size_r, size_c - new_size_c);
    add(C, ini_cr + new_size_r, ini_cc + new_size_c,
        T, ini_tr + new_size_r, ini_tc + new_size_c,
        size_r - new_size_r, size_c - new_size_c);
    return;
}

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

    Matrix T = new_matrix_clean(size_ar, size_bc);

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
                   T->matrix, 0, 0,
                   new_size_ar, size_ac - new_size_ac, new_size_bc, min_size);
    matmul_seq_rec(A, ini_ar, ini_ac + new_size_ac,
                   B, ini_br + new_size_ac, ini_bc + new_size_bc,
                   T->matrix, 0, new_size_bc,
                   new_size_ar, size_ac - new_size_ac, size_bc - new_size_bc, min_size);
    matmul_seq_rec(A, ini_ar + new_size_ar, ini_ac + new_size_ac,
                   B, ini_br + new_size_ac, ini_bc,
                   T->matrix, new_size_ar, 0,
                   size_ar - new_size_ar, size_ac - new_size_ac, new_size_bc, min_size);
    matmul_seq_rec(A, ini_ar + new_size_ar, ini_ac + new_size_ac,
                   B, ini_br + new_size_ac, ini_bc + new_size_bc,
                   T->matrix, new_size_ar, new_size_bc,
                   size_ar - new_size_ar, size_ac - new_size_ac, size_bc - new_size_bc, min_size);

    add(C, ini_cr, ini_cc, T->matrix, 0, 0, size_ar, size_bc);
    destroy_matrix(T);
    return ;
}

void matmul_seq(Matrix A, Matrix B, Matrix C, int min_size) {
    matmul_seq_rec(A->matrix, 0, 0, B->matrix, 0, 0, C->matrix, 0, 0, A->n, A->m, B->m, min_size);
}

void matmul_seq_rec2(double** A, int ini_ar, int ini_ac,
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

    //Matrix T = new_matrix_clean(size_ar, size_bc);

    matmul_seq_rec2(A, ini_ar, ini_ac,
                   B, ini_br, ini_bc,
                   C, ini_cr, ini_cc,
                   new_size_ar, new_size_ac, new_size_bc, min_size);
    matmul_seq_rec2(A, ini_ar, ini_ac,
                   B, ini_br, ini_bc + new_size_bc,
                   C, ini_cr, ini_cc + new_size_bc,
                   new_size_ar, new_size_ac, size_bc - new_size_bc, min_size);
    matmul_seq_rec2(A, ini_ar + new_size_ar, ini_ac,
                   B, ini_br, ini_bc,
                   C, ini_cr + new_size_ar, ini_cc,
                   size_ar - new_size_ar, new_size_ac, new_size_bc, min_size);
    matmul_seq_rec2(A, ini_ar + new_size_ar, ini_ac,
                   B, ini_br, ini_bc + new_size_bc,
                   C, ini_cr + new_size_ar, ini_cc + new_size_bc,
                   size_ar - new_size_ar, new_size_ac, size_bc - new_size_bc, min_size);

    matmul_seq_rec2(A, ini_ar, ini_ac + new_size_ac,
                   B, ini_br + new_size_ac, ini_bc,
                   C, ini_cr, ini_cc,
                   new_size_ar, size_ac - new_size_ac, new_size_bc, min_size);
    matmul_seq_rec2(A, ini_ar, ini_ac + new_size_ac,
                   B, ini_br + new_size_ac, ini_bc + new_size_bc,
                   C, ini_cr, ini_cc + new_size_bc,
                   new_size_ar, size_ac - new_size_ac, size_bc - new_size_bc, min_size);
    matmul_seq_rec2(A, ini_ar + new_size_ar, ini_ac + new_size_ac,
                   B, ini_br + new_size_ac, ini_bc,
                   C, ini_cr + new_size_ar, ini_cc,
                   size_ar - new_size_ar, size_ac - new_size_ac, new_size_bc, min_size);
    matmul_seq_rec2(A, ini_ar + new_size_ar, ini_ac + new_size_ac,
                   B, ini_br + new_size_ac, ini_bc + new_size_bc,
                   C, ini_cr + new_size_ar, ini_cc + new_size_bc,
                   size_ar - new_size_ar, size_ac - new_size_ac, size_bc - new_size_bc, min_size);

    //add(C, ini_cr, ini_cc, T->matrix, 0, 0, size_ar, size_bc);
    //destroy_matrix(T);
    return ;
}

void matmul_seq2(Matrix A, Matrix B, Matrix C, int min_size) {
    matmul_seq_rec2(A->matrix, 0, 0, B->matrix, 0, 0, C->matrix, 0, 0, A->n, A->m, B->m, min_size);
}
