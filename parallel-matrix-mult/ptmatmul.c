/*
 * @author: João Gabriel Basi Nº USP: 9793801
 * @author: Juliano Garcia de Oliveira Nº USP: 9277086
 *
 * MAC0219
 */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include "macros.h"
#include "ptmatmul.h"
#include "error.h"

void* matmul_pt_sub(void* arg) {
    Argument a = (Argument)arg;
    double* A = a->A;
    double* B = a->B;
    double* C = a->C;
    double *pB, *pC, r;
    uint64_t size_ac = a->size_ac;
    uint64_t size_ar = a->size_ar;
    uint64_t size_bc = a->size_bc;
    uint64_t or_size_ac = a->or_size_ac;
    uint64_t or_size_bc = a->or_size_bc;
    uint64_t i, j, k;
    for (i = 0; i < size_ar; i++) {
        for (k = 0; k < size_ac; k++) {
            r = *A++;
            pB = B + k*or_size_bc;
            pC = C + i*or_size_bc;
            for (j = 0; j < size_bc; j++)
                *pC++ += r* *pB++;
        }
        A += or_size_ac - size_ac;
    }
    free(a);
    return NULL;
}

void matmul_pt(MatrixArray A, MatrixArray B, MatrixArray C) {
    double *pA = A->m_c;
    double *pB = B->m_c;
    double *pC = C->m_c;
    uint64_t size_ar = A->n;
    uint64_t size_ac = A->m;
    uint64_t size_bc = B->m;
    uint64_t cacheSize = getCacheSize();
    uint64_t r_ar = ceil64(size_ar, ceil64(cacheSize, MIN_SIZE));
    uint64_t r_ac = ceil64(size_ac, MIN_SIZE);
    uint64_t r_bc = ceil64(size_bc, ceil64(cacheSize, MIN_SIZE));
    uint64_t i, j, k, diff_i, diff_j, diff_k, ini_i, ini_j, ini_k;
    pthread_t *t;
    t = emalloc(r_ar*r_bc*sizeof(pthread_t));
    for (i = 0, ini_i = 0; i < r_ac; i++) {
        diff_i = ceilDiff(i, size_ac, r_ac);
        for (k = 0, ini_k = 0; k < r_ar; k++) {
            diff_k = ceilDiff(k, size_ar, r_ar);
            for (j = 0, ini_j = 0; j < r_bc; j++) {
                diff_j = ceilDiff(j, size_bc, r_bc);
                Argument a = create_argument(pA + ini_k*size_ac + ini_i,
                                             pB + ini_i*size_bc + ini_j,
                                             pC + ini_k*size_bc + ini_j,
                                             diff_k, diff_i, diff_j,
                                             size_ac, size_bc, 0);
                pthread_create(t + k*r_bc + j, NULL, &matmul_pt_sub, (void*)a);
                ini_j += diff_j;
            }
            ini_k += diff_k;
        }
        for (k = 0; k < r_ar*r_bc; k++)
            pthread_join(t[k], NULL);
        ini_i += diff_i;
    }
    free(t);
}
