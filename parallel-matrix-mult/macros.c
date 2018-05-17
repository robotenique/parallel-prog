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
