#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "error.h"
#include "macros.h"

int IMPL_TYPE;

void matmul_seq(Matrix A, int ini_ar, int ini_ac,
                Matrix B, int ini_br, int ini_bc,
                Matrix C, int size_ar, int size_ac, int size_bc) {
    if (size_ar == 1 && size_ac == 1 && size_bc == 1) {
        C->matrix[ini_ar][ini_bc] += A->matrix[ini_ar][ini_ac]*B->matrix[ini_br][ini_bc];
        return ;
    }
    int new_size_ar = size_ar/2;
    int new_size_ac = size_ac/2;
    int new_size_bc = size_bc/2;
    matmul_seq(A, ini_ar, ini_ac, B, ini_br, ini_bc,
               C, new_size_ar, new_size_ac, new_size_bc);
    matmul_seq(A, ini_ar, ini_ac, B, ini_br, ini_bc + new_size_bc,
               C, new_size_ar, new_size_ac, size_bc - new_size_bc);
    matmul_seq(A, ini_ar + new_size_ar, ini_ac, B, ini_br, ini_bc,
               C, size_ar - new_size_ar, new_size_ac, new_size_bc);
    matmul_seq(A, ini_ar + new_size_ar, ini_ac, B, ini_br, ini_bc + new_size_bc,
               C, size_ar - new_size_ar, new_size_ac, size_bc - new_size_bc);

    // Originally this 4 lines are made using a temporary matrix T instead of C
    matmul_seq(A, ini_ar, ini_ac + new_size_ac, B, ini_br + new_size_ac, ini_bc,
               C, new_size_ar, size_ac - new_size_ac, new_size_bc);
    matmul_seq(A, ini_ar, ini_ac + new_size_ac, B, ini_br + new_size_ac, ini_bc + new_size_bc,
               C, new_size_ar, size_ac - new_size_ac, size_bc - new_size_bc);
    matmul_seq(A, ini_ar + new_size_ar, ini_ac + new_size_ac, B, ini_br + new_size_ac, ini_bc,
               C, size_ar - new_size_ar, size_ac - new_size_ac, new_size_bc);
    matmul_seq(A, ini_ar + new_size_ar, ini_ac + new_size_ac, B, ini_br + new_size_ac, ini_bc + new_size_bc,
               C, size_ar - new_size_ar, size_ac - new_size_ac, size_bc - new_size_bc);

    // With the temporary matrix T, C and T are added up recursively here
    // add(C, T);
    // free(T);
    return ;
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
    print_matrix(mtx_A);
    print_matrix(mtx_B);
    matmul_seq(mtx_A, 0, 0, mtx_B, 0, 0, mtx_C, mtx_A->n, mtx_A->m, mtx_B->m);
    print_matrix(mtx_C);
    // write_matrix(mtx_C);

}
