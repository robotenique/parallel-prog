/*
 * @author: João Gabriel Basi Nº USP: 9793801
 * @author: Juliano Garcia de Oliveira Nº USP: 9277086
 *
 * MAC0219
 */

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

    //double dtime;

    if(argc < 5)
        die("Wrong number of arguments!\nUsage ./main <impl> <file_matrixA> <file_matrixB> <file_matrixC>");

    if (!strcmp(argv[1], "p"))
        IMPL_TYPE = 0;
    else if (!strcmp(argv[1], "o"))
        IMPL_TYPE = 1;
    else
        die("Invalid <implementation> provided.");

    char *file_A = estrdup(argv[2]);
    char *file_B = estrdup(argv[3]);
    char *file_C = estrdup(argv[4]);

    if (IMPL_TYPE == 0) {
        /* -------------- Using PThreads -------------- */
        MatrixArray mtxArr_A = new_matrixArray(file_A);
        MatrixArray mtxArr_B = new_matrixArray(file_B);
        MatrixArray mtxArr_C = new_matrixArray_clean(mtxArr_A->n, mtxArr_B->m);

        //dtime = omp_get_wtime();
        matmul_pt(mtxArr_A, mtxArr_B, mtxArr_C);
        //dtime = omp_get_wtime() - dtime;

        //printf("%ld;%ld;%f\n", mtxArr_A->n, mtxArr_B->n, dtime);

        write_matrixArray(mtxArr_C, file_C);

        destroy_matrixArray(mtxArr_A);
        destroy_matrixArray(mtxArr_B);
        destroy_matrixArray(mtxArr_C);

        free(file_A);
        free(file_B);
        free(file_C);

        return 0;
    }
    if (IMPL_TYPE == 1) {
        /* -------------- Using OpenMP -------------- */
        MatrixArray mtxArr_A = new_matrixArray(file_A);
        MatrixArray mtxArr_B = new_matrixArray(file_B);
        MatrixArray mtxArr_C = new_matrixArray_clean(mtxArr_A->n, mtxArr_B->m);

        //dtime = omp_get_wtime();
        matmul_omp(mtxArr_A, mtxArr_B, mtxArr_C);
        //dtime = omp_get_wtime() - dtime;

        //printf("%ld;%ld;%f\n", mtxArr_A->n, mtxArr_B->n, dtime);

        write_matrixArray(mtxArr_C, file_C);

        destroy_matrixArray(mtxArr_A);
        destroy_matrixArray(mtxArr_B);
        destroy_matrixArray(mtxArr_C);

        free(file_A);
        free(file_B);
        free(file_C);

        return 0;
    }

    return 0;
}
