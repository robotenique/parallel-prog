/*
 * @author: João Gabriel Basi Nº USP: 9793801
 * @author: Juliano Garcia de Oliveira Nº USP: 9277086
 *
 * MAC0219
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>
#include "macros.h"
#include "error.h"

MatrixArray new_matrixArray(char* filename){
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    char * pch;
    size_t read;

    uint64_t curr_i, curr_j;
    MatrixArray mtx = (MatrixArray)emalloc(sizeof(mat_c));
    fp = efopen(filename, "r");
    uint64_t lnum = 0;
    while ((read = getline(&line, &len, fp)) != -1) {
        if(lnum == 0){
            pch = strtok (line," ");
            mtx->n = (uint64_t)atoi(pch);
            pch = strtok (NULL, " ");
            mtx->m = (uint64_t)atoi(pch);
            mtx->nm = mtx->n*mtx->m;
            mtx->m_c = calloc(mtx->nm, sizeof(double));
            if (mtx->m_c == NULL)
                die("Failed to allocate memory");

        }
        else{
            pch = strtok (line," ");
            curr_i = (uint64_t)atoi(pch);
            pch = strtok (NULL, " ");
            curr_j = (uint64_t)atoi(pch);
            pch = strtok (NULL, " ");
            mtx->m_c[(curr_i - 1)*mtx->m + (curr_j - 1)] = atof(pch);
        }
        lnum++;
    }

    fclose(fp);
    if (line)
        free(line);
    return mtx;
}
MatrixArray new_matrixArray_clean(uint64_t n, uint64_t m){
    MatrixArray mtx = (MatrixArray)emalloc(sizeof(mat_c));
    mtx->n = n;
    mtx->m = m;
    mtx->nm = n*m;
    mtx->m_c = calloc(mtx->nm, sizeof(double));
    return mtx;
}
void destroy_matrixArray(MatrixArray mtxArr){
    free(mtxArr->m_c);
    free(mtxArr);
}
void write_matrixArray(MatrixArray mtxArr, char* filename){
    FILE *fp = efopen(filename, "w");
    fprintf(fp, "%lu %lu\n", mtxArr->n, mtxArr->m);
    for (uint64_t i = 0; i < mtxArr->n; i++)
        for (uint64_t j = 0; j < mtxArr->m; j++)
            if(mtxArr->m_c[i*mtxArr->m + j] != 0)
                fprintf(fp, "%lu %lu %lf\n", i + 1, j + 1, mtxArr->m_c[i*mtxArr->n + j]);

    fclose(fp);
}
void reset_matrixArray(MatrixArray mtxArr){
    memset(mtxArr->m_c, 0.0f, mtxArr->nm*sizeof(double));
}

Argument create_argument(double* A, double* B, double* C,
                         uint64_t size_ar, uint64_t size_ac, uint64_t size_bc,
                         uint64_t or_size_ac, uint64_t or_size_bc,
                         uint64_t num_threads) {
    Argument a = emalloc(sizeof(targ));
    a->A = A;
    a->B = B;
    a->C = C;
    a->size_ar = size_ar;
    a->size_ac = size_ac;
    a->size_bc = size_bc;
    a->or_size_ac = or_size_ac;
    a->or_size_bc = or_size_bc;
    a->num_threads = num_threads;
    return a;
}

uint64_t ceil64(uint64_t num, uint64_t den) {
    if (!num) return 0;
    return (num-1)/den + 1;
}

uint64_t ceilDiff(uint64_t coef, uint64_t num, uint64_t den) {
    return ceil64((coef+1)*num, den) - ceil64(coef*num, den);
}

uint64_t getCacheSize() {
    return sysconf(_SC_LEVEL1_DCACHE_SIZE);
}
