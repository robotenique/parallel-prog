#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "error.h"
#include "macros.h"

int IMPL_TYPE;

void add(Matrix C, int ini_cr, int ini_cc,
         Matrix T, int ini_tr, int ini_tc,
         int size_r, int size_c) {
    if (!size_r || !size_c)
        return ;
    if (size_r == 1 && size_c == 1) {
        C->matrix[ini_cr][ini_cc] += T->matrix[ini_tr][ini_tc];
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

void matmul_seq_rec(Matrix A, int ini_ar, int ini_ac,
                    Matrix B, int ini_br, int ini_bc,
                    Matrix C, int ini_cr, int ini_cc,
                    int size_ar, int size_ac, int size_bc) {
    if (!size_ac || !size_ar || !size_bc)
        return ;
    if (size_ar == 1 && size_ac == 1 && size_bc == 1) {
        C->matrix[ini_cr][ini_cc] = A->matrix[ini_ar][ini_ac]*B->matrix[ini_br][ini_bc];
        return ;
    }
    int new_size_ar = size_ar/2;
    int new_size_ac = size_ac/2;
    int new_size_bc = size_bc/2;

    Matrix T = new_matrix_clean(size_ar, size_bc);

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

    // Originally this 4 lines are made using a temporary matrix T instead of C
    matmul_seq_rec(A, ini_ar, ini_ac + new_size_ac,
                   B, ini_br + new_size_ac, ini_bc,
                   T, 0, 0,
                   new_size_ar, size_ac - new_size_ac, new_size_bc);
    matmul_seq_rec(A, ini_ar, ini_ac + new_size_ac,
                   B, ini_br + new_size_ac, ini_bc + new_size_bc,
                   T, 0, new_size_bc,
                   new_size_ar, size_ac - new_size_ac, size_bc - new_size_bc);
    matmul_seq_rec(A, ini_ar + new_size_ar, ini_ac + new_size_ac,
                   B, ini_br + new_size_ac, ini_bc,
                   T, new_size_ar, 0,
                   size_ar - new_size_ar, size_ac - new_size_ac, new_size_bc);
    matmul_seq_rec(A, ini_ar + new_size_ar, ini_ac + new_size_ac,
                   B, ini_br + new_size_ac, ini_bc + new_size_bc,
                   T, new_size_ar, new_size_bc,
                   size_ar - new_size_ar, size_ac - new_size_ac, size_bc - new_size_bc);

    // With the temporary matrix T, C and T are added up recursively here
    add(C, ini_cr, ini_cc, T, 0, 0);
    // free(T);
    return ;
}

void matmul_seq(Matrix A, Matrix B, Matrix C) {
    matmul_seq_rec(A, 0, 0, B, 0, 0, C, 0, 0, A->n, A->m, B->m);
}

int main(int argc, char const *argv[]) {
    set_prog_name("matmul");
    if(argc < 5)
        die("Wrong number of arguments!\nUsage ./main <impl> <file_matrixA> <file_matrixB> <file_matrixC>");

    if (!strcmp(argv[1], "p"))
        IMPL_TYPE = 0;
    else if(!strcmp(argv[1], "o"))
        IMPL_TYPE = 1;
    else
        die("Invalid <implementation> provided.");
    char *file_A = estrdup(argv[2]);
    char *file_B = estrdup(argv[3]);
    char *file_C = estrdup(argv[4]);

    Matrix mtx_A = new_matrix(file_A);
    Matrix mtx_B = new_matrix(file_B);
    Matrix mtx_C = new_matrix_clean(mtx_A->n, mtx_B->m);
    //print_matrix(mtx_A);
    //print_matrix(mtx_B);
    matmul_seq(mtx_A, mtx_B, mtx_C);
    print_matrix(mtx_C);
    matmul_seq(mtx_A, mtx_C, mtx_B);
    print_matrix(mtx_B);
    matmul_seq(mtx_A, mtx_B, mtx_C);
    print_matrix(mtx_C);

    // write_matrix(mtx_C);

}
