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
#include "macros.h"
#include "error.h"

void print_matrix(Matrix mtx) {
    printf("\n");
    for (int i = 0; i < mtx->n; i++) {
        for (int j = 0; j < mtx->m - 1; j++) {
            printf("%lf ", mtx->matrix[i][j]);
        }
        printf("%lf\n", mtx->matrix[i][mtx->m - 1]);
    }
}



Matrix new_matrix(char* filename) {
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    char * pch;
    size_t read;

    int curr_i, curr_j;

    Matrix mtx = (Matrix)emalloc(sizeof(mat));

    fp = efopen(filename, "r");
    u_int lnum = 0;
    while ((read = getline(&line, &len, fp)) != -1) {
        if(lnum == 0){
            pch = strtok (line," ");
            mtx->n = (u_int)atoi(pch);
            pch = strtok (NULL, " ");
            mtx->m = (u_int)atoi(pch);
            mtx->matrix = emalloc(mtx->n*sizeof(double*));
            for (int i = 0; i < mtx->n; i++) {
                mtx->matrix[i] = calloc(mtx->m, sizeof(double));
                if (mtx->matrix[i] == NULL)
                    die("Failed to allocate memory");
            }
        }
        else{
            pch = strtok (line," ");
            curr_i = (u_int)atoi(pch);
            pch = strtok (NULL, " ");
            curr_j = (u_int)atoi(pch);
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

Matrix new_matrix_clean(u_int n, u_int m) {
    Matrix mtx = (Matrix)emalloc(sizeof(mat));
    mtx->matrix = emalloc(n*sizeof(double*));
    for (int i = 0; i < n; i++) {
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
    fprintf(fp, "%u %u\n", mtx->n, mtx->m);
    for (int i = 0; i < mtx->n; i++)
        for (int j = 0; j < mtx->m; j++)
            if(mtx->matrix[i][j] != 0)
                fprintf(fp, "%d %d %lf\n", i + 1, j + 1, mtx->matrix[i][j]);

    fclose(fp);
}

void reset_matrix(Matrix mtx) {
    for (int i = 0; i < mtx->n; i++)
        for (int j = 0; j < mtx->m; j++)
            mtx->matrix[i][j] = 0.0f;
}

void destroy_matrix(Matrix mtx) {
    for (int i = 0; i < mtx->n; i++)
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

    int curr_i, curr_j;
    MatrixArray mtx = (MatrixArray)emalloc(sizeof(mat_c));
    fp = efopen(filename, "r");
    u_int lnum = 0;
    while ((read = getline(&line, &len, fp)) != -1) {
        if(lnum == 0){
            pch = strtok (line," ");
            mtx->n = (u_int)atoi(pch);
            pch = strtok (NULL, " ");
            mtx->m = (u_int)atoi(pch);
            mtx->nm = mtx->n*mtx->m;
            mtx->m_c = calloc(mtx->nm, sizeof(double));
            if (mtx->m_c == NULL)
                die("Failed to allocate memory");

        }
        else{
            pch = strtok (line," ");
            // printf("PCH %s  ", pch);
            curr_i = (u_int)atoi(pch);
            pch = strtok (NULL, " ");
            // printf("PCH %s  ", pch);
            curr_j = (u_int)atoi(pch);
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
MatrixArray new_matrixArray_clean(u_int n, u_int m){
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
    fprintf(fp, "%u %u\n", mtxArr->n, mtxArr->m);
    for (int i = 0; i < mtxArr->n; i++)
        for (int j = 0; j < mtxArr->m; j++)
            if(mtxArr->m_c[i*mtxArr->m + j] != 0)
                fprintf(fp, "%d %d %lf\n", i + 1, j + 1, mtxArr->m_c[i*mtxArr->n + j]);

    fclose(fp);
}
void reset_matrixArray(MatrixArray mtxArr){
    memset(mtxArr->m_c, 0.0f, mtxArr->nm*sizeof(double));
}
void print_matrixArray(MatrixArray mtxArr) {
    printf("\n");
    for (int i = 0; i < mtxArr->n; i++) {
        for (int j = 0; j < mtxArr->m - 1; j++) {
            printf("%lf ", mtxArr->m_c[i*mtxArr->m + j]);
        }
        printf("%lf\n", mtxArr->m_c[i*mtxArr->m + (mtxArr->m - 1)]);
    }
}

bool are_equal_ma2m(MatrixArray ma, Matrix m){
    for (int i = 0; i < ma->n; i++)
        for (int j = 0; j < ma->m; j++)
            if(ma->m_c[i*ma->m + j] != m->matrix[i][j]){
                printf("deu diferente %d %d , %lf , %lf\n",i, j,ma->m_c[i*ma->m + j], m->matrix[i][j]);
                return false;
            }

    return true;
}

Argument create_argument(double** A, u_int ini_ar, u_int ini_ac,
                         double** B, u_int ini_br, u_int ini_bc,
                         double** C, u_int ini_cr, u_int ini_cc,
                         u_int size_ar, u_int size_ac, u_int size_bc, u_int min_size) {
        Argument a = emalloc(sizeof(targ));
        a->A = A;
        a->B = B;
        a->C = C;
        a->ini_ar = ini_ar;
        a->ini_ac = ini_ac;
        a->ini_br = ini_br;
        a->ini_bc = ini_bc;
        a->ini_cr = ini_cr;
        a->ini_cc = ini_cc;
        a->size_ar = size_ar;
        a->size_ac = size_ac;
        a->size_bc = size_bc;
        a->min_size = min_size;
        return a;
    }
