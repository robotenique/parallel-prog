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
#include <inttypes.h>
#include <unistd.h> // TODO: Remove this header
#include "macros.h"
#include "error.h"

uint64_t ceil64(uint64_t num, uint64_t den) {
    if (!num) return 0;
    return (num-1)/den + 1;
}

uint64_t ceilDiff(uint64_t coef, uint64_t num, uint64_t den) {
    return ceil64((coef+1)*num, den) - ceil64(coef*num, den);
}

Matrix new_matrix(char* filename) {
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    char * pch;
    size_t read;

    uint64_t curr_i, curr_j;

    Matrix mtx = (Matrix)emalloc(sizeof(mat));

    fp = efopen(filename, "r");
    uint64_t lnum = 0;
    while ((read = getline(&line, &len, fp)) != -1) {
        if(lnum == 0){
            pch = strtok (line," ");
            mtx->n = (uint64_t)atoi(pch);
            pch = strtok (NULL, " ");
            mtx->m = (uint64_t)atoi(pch);
            mtx->matrix = emalloc(mtx->n*sizeof(double*));
            for (uint64_t i = 0; i < mtx->n; i++) {
                mtx->matrix[i] = calloc(mtx->m, sizeof(double));
                if (mtx->matrix[i] == NULL)
                    die("Failed to allocate memory");
            }
        }
        else{
            pch = strtok (line," ");
            curr_i = (uint64_t)atoi(pch);
            pch = strtok (NULL, " ");
            curr_j = (uint64_t)atoi(pch);
            pch = strtok (NULL, " ");
            mtx->matrix[curr_i - 1][curr_j - 1] = atof(pch);
        }
        lnum++;
    }

    fclose(fp);
    if (line)
        free(line);
    return mtx;
}

Matrix new_matrix_clean(uint64_t n, uint64_t m) {
    Matrix mtx = (Matrix)emalloc(sizeof(mat));
    mtx->matrix = emalloc(n*sizeof(double*));
    for (uint64_t i = 0; i < n; i++) {
        mtx->matrix[i] = calloc(m, sizeof(double));
        if (mtx->matrix[i] == NULL)
            die("Failed to allocate memory");
    }
    mtx->n = n;
    mtx->m = m;
    return mtx;
}

void write_matrix(Matrix mtx, char* filename){
    FILE *fp = efopen(filename, "w");
    fprintf(fp, "%lu %lu\n", mtx->n, mtx->m);
    for (uint64_t i = 0; i < mtx->n; i++)
        for (uint64_t j = 0; j < mtx->m; j++)
            if(mtx->matrix[i][j] != 0)
                fprintf(fp, "%lu %lu %lf\n", i + 1, j + 1, mtx->matrix[i][j]);

    fclose(fp);
}

void reset_matrix(Matrix mtx) {
    for (uint64_t i = 0; i < mtx->n; i++)
        for (uint64_t j = 0; j < mtx->m; j++)
            mtx->matrix[i][j] = 0.0f;
}

void destroy_matrix(Matrix mtx) {
    for (uint64_t i = 0; i < mtx->n; i++)
        free(mtx->matrix[i]);
    free(mtx->matrix);
    free(mtx);
}

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
            // printf("PCH %s  ", pch);
            curr_i = (uint64_t)atoi(pch);
            pch = strtok (NULL, " ");
            // printf("PCH %s  ", pch);
            curr_j = (uint64_t)atoi(pch);
            pch = strtok (NULL, " ");
            // printf("PCH %s  ", pch);
            mtx->m_c[(curr_i - 1)*mtx->m + (curr_j - 1)] = atof(pch);
            // printf("TERMINEI!\n");
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

void print_matrix(Matrix mtx) {
    printf("\n");
    for (uint64_t i = 0; i < mtx->n; i++) {
        for (uint64_t j = 0; j < mtx->m - 1; j++) {
            printf("%lf ", mtx->matrix[i][j]);
        }
        printf("%lf\n", mtx->matrix[i][mtx->m - 1]);
    }
}

void print_matrixArray(MatrixArray mtxArr) {
    printf("\n");
    for (uint64_t i = 0; i < mtxArr->n; i++) {
        for (uint64_t j = 0; j < mtxArr->m - 1; j++) {
            printf("%lf ", mtxArr->m_c[i*mtxArr->m + j]);
        }
        printf("%lf\n", mtxArr->m_c[i*mtxArr->m + (mtxArr->m - 1)]);
    }
}

bool are_equal_ma2m(MatrixArray ma, Matrix m){
    for (uint64_t i = 0; i < ma->n; i++)
        for (uint64_t j = 0; j < ma->m; j++)
            if(ma->m_c[i*ma->m + j] != m->matrix[i][j]){
                printf("deu diferente %lu %lu , %lf , %lf\n",i, j,ma->m_c[i*ma->m + j], m->matrix[i][j]);
                return false;
            }

    return true;
}

bool are_equal_ma2ma(MatrixArray ma, MatrixArray m){
    for (uint64_t i = 0; i < ma->n; i++)
        for (uint64_t j = 0; j < ma->m; j++)
            if(ma->m_c[i*ma->m + j] != m->m_c[i*m->m + j]){
                printf("deu diferente %lu %lu , %lf , %lf\n",i, j,ma->m_c[i*ma->m + j], m->m_c[i*m->m + j]);
                return false;
            }

    return true;
}

void print_num_threads() {
    char s[80];
    sprintf(s, "cat /proc/%d/status | grep \"Threads\" | tr -d \"Threads:\"", getpid());
    FILE *p = popen(s, "r");
    int nt;
    if (p != NULL) {
        fscanf(p, "%d", &nt);
        printf("%d\n", nt);
    }
}

uint64_t getCacheSize() {
    return sysconf(_SC_LEVEL1_DCACHE_SIZE);
}
