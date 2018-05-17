#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "error.h"
#include "macros.h"

int IMPL_TYPE;



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
    // write_matrix(mtx_C);

}
