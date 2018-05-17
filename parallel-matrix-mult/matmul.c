#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "error.h"
#include "macros.h"
#include "seqmatmul.h"

#define BILLION 1000000000.0
#define MILLION 1000000.0
#define HUNDRED 1000.0

int IMPL_TYPE;

int main(int argc, char const *argv[]) {
    set_prog_name("matmul");

    struct timespec start, finish;
    double elapsed;

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
    Matrix mtx_C;

    printf("time\n");

    for (int min_size = 64; min_size <= 2048; min_size += 64) {
        mtx_C = new_matrix_clean(mtx_A->n, mtx_B->m);
        // print_matrix(mtx_A);
        // print_matrix(mtx_B);

        clock_gettime(CLOCK_MONOTONIC, &start);

        matmul_seq_opt(mtx_A, mtx_B, mtx_C, min_size);

        clock_gettime(CLOCK_MONOTONIC, &finish);

        elapsed = (finish.tv_sec - start.tv_sec) * HUNDRED;
        elapsed += (finish.tv_nsec - start.tv_nsec) / MILLION;
        printf("%f\n", elapsed);

        //print_matrix(mtx_C);
    }

    //write_matrix(mtx_C, file_C);
}
