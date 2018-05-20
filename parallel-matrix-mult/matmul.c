#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <inttypes.h>
#include "error.h"
#include "macros.h"
#include "seqmatmul.h"
#include "openmpmatmul.h"
#include "ptmatmul.h"

int IMPL_TYPE;

int main(int argc, char const *argv[]) {
    set_prog_name("matmul");

    double dtime;

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
    /* -------------- Using openMP -------------- */
    MatrixArray mtxArr_A = new_matrixArray(file_A);
    MatrixArray mtxArr_B = new_matrixArray(file_B);
    MatrixArray refMtxArr_C = new_matrixArray_clean(mtxArr_A->n, mtxArr_B->m);
    printf("MATRIZ A:\n");
    //print_matrixArray(mtxArr_A);
    printf("MATRIZ B:\n");
    //print_matrixArray(mtxArr_B);
    printf("--\n");

    dtime = omp_get_wtime();
    matmul_omp(mtxArr_A, mtxArr_B, refMtxArr_C);
    //matcoisa(mtxArr_A, mtxArr_B, mtxArr_C);
    dtime = omp_get_wtime() - dtime;

    printf("\n --- Parallel matmul (omp) ---\n");
    //print_matrixArray(refMtxArr_C);
    printf("Elapsed: %f\n", dtime);

    // exit(1);

    printf("\n --- Parallel matmul (pt) ---\n");
    MatrixArray mtxArr_C = new_matrixArray_clean(mtxArr_A->n, mtxArr_B->m);

    dtime = omp_get_wtime();
    //matmul_omp_2(mtx_A, mtx_B, mtx_C);
    matmul_pt(mtxArr_A, mtxArr_B, mtxArr_C, 256);
    dtime = omp_get_wtime() - dtime;

    //print_matrixArray(mtxArr_C);
    printf("Elapsed: %f\n", dtime);

    if(are_equal_ma2ma(refMtxArr_C, mtxArr_C))
        printf("WAU!!!\n");
    else
        printf("LIXOOOOOOOOOOO\n");

    // exit(1);

    printf("\n --- Seq matmul ---\n");

    Matrix mtx_A = new_matrix(file_A);
    Matrix mtx_B = new_matrix(file_B);
    Matrix mtx_C = new_matrix_clean(mtx_A->n, mtx_B->m);

    dtime = omp_get_wtime();
    matmul_seq(mtx_A, mtx_B, mtx_C, MIN_SIZE);
    //matmul_trashy(mtx_A, mtx_B, mtx_C);
    dtime = omp_get_wtime() - dtime;

    //print_matrix(mtx_C);
    printf("Elapsed: %f\n", dtime);

    if(are_equal_ma2m(refMtxArr_C, mtx_C))
        printf("WAU!!!\n");
    else
        printf("LIXOOOOOOOOOOO\n");

    //exit(1);

    destroy_matrixArray(mtxArr_A);
    destroy_matrixArray(mtxArr_B);
    destroy_matrixArray(mtxArr_C);
    destroy_matrixArray(refMtxArr_C);

    destroy_matrix(mtx_A);
    destroy_matrix(mtx_B);
    destroy_matrix(mtx_C);

    //exit(1);

    //write_matrix(mtx_C, file_C);
    return 0;
}
