#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include "macros.h"
#include "ptmatmul.h"
#include "error.h"

void* matmul_pt_rec(void* arg) {
    Argument a = (Argument)arg;
    if (!(a->size_ac) || !(a->size_ar) || !(a->size_bc)) {
        free(a);
        return NULL;
    }
    double* A = a->A;
    double* B = a->B;
    double* C = a->C;
    uint64_t size_ac = a->size_ac;
    uint64_t size_ar = a->size_ar;
    uint64_t size_bc = a->size_bc;
    uint64_t or_size_ac = a->or_size_ac;
    uint64_t or_size_bc = a->or_size_bc;
    uint64_t num_threads = a->num_threads;
    if (num_threads >= MAX_THREADS ||
        (size_ar <= MIN_SIZE && size_ac <= MIN_SIZE && size_bc <= MIN_SIZE)) {
        uint64_t i, j, k;
        double *pB, *pC, r;
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
    uint64_t new_size_ar = size_ar/2;
    uint64_t new_size_ac = size_ac/2;
    uint64_t new_size_bc = size_bc/2;
    pthread_t t1, t2, t3, t4, t5, t6, t7, t8;
    Argument a_tmp;

    a_tmp = create_argument(A, B, C,
                  new_size_ar, new_size_ac, new_size_bc,
                  or_size_ac, or_size_bc,
                  4*num_threads);
    pthread_create(&t1, NULL, &matmul_pt_rec, (void*)a_tmp);
    a_tmp = create_argument(A, B + new_size_bc, C + new_size_bc,
                  new_size_ar, new_size_ac, size_bc - new_size_bc,
                  or_size_ac, or_size_bc,
                  4*num_threads);
    pthread_create(&t2, NULL, &matmul_pt_rec, (void*)a_tmp);
    a_tmp = create_argument(A + new_size_ar*or_size_ac, B, C + new_size_ar*or_size_bc,
                  size_ar - new_size_ar, new_size_ac, new_size_bc,
                  or_size_ac, or_size_bc,
                  4*num_threads);
    pthread_create(&t3, NULL, &matmul_pt_rec, (void*)a_tmp);
    a_tmp = create_argument(A + new_size_ar*or_size_ac, B + new_size_bc,
                  C + new_size_ar*or_size_bc + new_size_bc,
                  size_ar - new_size_ar, new_size_ac, size_bc - new_size_bc,
                  or_size_ac, or_size_bc,
                  4*num_threads);
    pthread_create(&t4, NULL, &matmul_pt_rec, (void*)a_tmp);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    pthread_join(t3, NULL);
    pthread_join(t4, NULL);

    a_tmp = create_argument(A + new_size_ac, B + new_size_ac*or_size_bc, C,
                  new_size_ar, size_ac - new_size_ac, new_size_bc,
                  or_size_ac, or_size_bc,
                  4*num_threads);
    pthread_create(&t5, NULL, &matmul_pt_rec, (void*)a_tmp);
    a_tmp = create_argument(A + new_size_ac, B + new_size_ac*or_size_bc + new_size_bc,
                  C + new_size_bc,
                  new_size_ar, size_ac - new_size_ac, size_bc - new_size_bc,
                  or_size_ac, or_size_bc,
                  4*num_threads);
    pthread_create(&t6, NULL, &matmul_pt_rec, (void*)a_tmp);
    a_tmp = create_argument(A + new_size_ar*or_size_ac + new_size_ac, B + new_size_ac*or_size_bc,
                  C + new_size_ar*or_size_bc,
                  size_ar - new_size_ar, size_ac - new_size_ac, new_size_bc,
                  or_size_ac, or_size_bc,
                  4*num_threads);
    pthread_create(&t7, NULL, &matmul_pt_rec, (void*)a_tmp);
    a_tmp = create_argument(A + new_size_ar*or_size_ac + new_size_ac,
                  B + new_size_ac*or_size_bc + new_size_bc,
                  C + new_size_ar*or_size_bc + new_size_bc,
                  size_ar - new_size_ar, size_ac - new_size_ac, size_bc - new_size_bc,
                  or_size_ac, or_size_bc,
                  4*num_threads);
    pthread_create(&t8, NULL, &matmul_pt_rec, (void*)a_tmp);

    pthread_join(t5, NULL);
    pthread_join(t6, NULL);
    pthread_join(t7, NULL);
    pthread_join(t8, NULL);

    free(a);
    return NULL;
}

void matmul_pt(MatrixArray A, MatrixArray B, MatrixArray C) {
    Argument a = create_argument(A->m_c, B->m_c, C->m_c, A->n, A->m, B->m,
                                 A->m, B->m, 1);
    matmul_pt_rec((void*)a);
}

void* matmul_pt_rec2(void* arg) {
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

void matmul_pt2(MatrixArray A, MatrixArray B, MatrixArray C) {
    double *pA = A->m_c;
    double *pB = B->m_c;
    double *pC = C->m_c;
    uint64_t size_ar = A->n;
    uint64_t size_ac = A->m;
    uint64_t size_bc = B->m;
    uint64_t r_ar = ceil64(size_ar, MIN_SIZE);
    uint64_t r_ac = ceil64(size_ac, MIN_SIZE);
    uint64_t r_bc = ceil64(size_bc, MIN_SIZE);
    uint64_t i, j, k, diff_i, diff_j, diff_k, ini_i, ini_j, ini_k;
    pthread_t *t;
    printf("Num threads: %ld\n", r_ar*r_bc);
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
                pthread_create(t + k*r_bc + j, NULL, &matmul_pt_rec2, (void*)a);
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
