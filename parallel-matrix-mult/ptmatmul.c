#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <unistd.h>
#include "macros.h"
#include "ptmatmul.h"

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

void* matmul_pt_rec(void* arg) {
    Argument a = (Argument)arg;
    if (!(a->size_ac) || !(a->size_ar) || !(a->size_bc)) {
        free(a);
        return NULL;
    }
    //print_num_threads();
    double** A = a->A;
    double** B = a->B;
    double** C = a->C;
    uint64_t size_ac = a->size_ac;
    uint64_t size_ar = a->size_ar;
    uint64_t size_bc = a->size_bc;
    uint64_t ini_ar = a->ini_ar;
    uint64_t ini_ac = a->ini_ac;
    uint64_t ini_br = a->ini_br;
    uint64_t ini_bc = a->ini_bc;
    uint64_t ini_cr = a->ini_cr;
    uint64_t ini_cc = a->ini_cc;
    uint64_t min_size = a->min_size;
    uint64_t num_threads = a->num_threads;
    if (num_threads >= MAX_THREADS ||
        (size_ar <= min_size && size_ac <= min_size && size_bc <= min_size)) {
        for (uint64_t i = 0; i < size_ar; i++) {
            for (uint64_t k = 0; k < size_ac; k++) {
                for (uint64_t j = 0; j < size_bc; j++)
                    C[ini_cr+i][ini_cc+j] += A[ini_ar+i][ini_ac+k]*B[ini_br+k][ini_bc+j];
            }
        }
        free(a);
        return NULL;
    }
    uint64_t new_size_ar = size_ar/2;
    uint64_t new_size_ac = size_ac/2;
    uint64_t new_size_bc = size_bc/2;
    pthread_t t1, t2, t3, t4, t5, t6, t7, t8;
    Argument a_tmp;

    a_tmp = create_argument(A, ini_ar, ini_ac,
                  B, ini_br, ini_bc,
                  C, ini_cr, ini_cc,
                  new_size_ar, new_size_ac, new_size_bc,
                  min_size, 4*num_threads);
    pthread_create(&t1, NULL, &matmul_pt_rec, (void*)a_tmp);
    a_tmp = create_argument(A, ini_ar, ini_ac,
                  B, ini_br, ini_bc + new_size_bc,
                  C, ini_cr, ini_cc + new_size_bc,
                  new_size_ar, new_size_ac, size_bc - new_size_bc,
                  min_size, 4*num_threads);
    pthread_create(&t2, NULL, &matmul_pt_rec, (void*)a_tmp);
    a_tmp = create_argument(A, ini_ar + new_size_ar, ini_ac,
                  B, ini_br, ini_bc,
                  C, ini_cr + new_size_ar, ini_cc,
                  size_ar - new_size_ar, new_size_ac, new_size_bc,
                  min_size, 4*num_threads);
    pthread_create(&t3, NULL, &matmul_pt_rec, (void*)a_tmp);
    a_tmp = create_argument(A, ini_ar + new_size_ar, ini_ac,
                  B, ini_br, ini_bc + new_size_bc,
                  C, ini_cr + new_size_ar, ini_cc + new_size_bc,
                  size_ar - new_size_ar, new_size_ac, size_bc - new_size_bc,
                  min_size, 4*num_threads);
    pthread_create(&t4, NULL, &matmul_pt_rec, (void*)a_tmp);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    pthread_join(t3, NULL);
    pthread_join(t4, NULL);

    a_tmp = create_argument(A, ini_ar, ini_ac + new_size_ac,
                  B, ini_br + new_size_ac, ini_bc,
                  C, ini_cr, ini_cc,
                  new_size_ar, size_ac - new_size_ac, new_size_bc,
                  min_size, 4*num_threads);
    pthread_create(&t5, NULL, &matmul_pt_rec, (void*)a_tmp);
    a_tmp = create_argument(A, ini_ar, ini_ac + new_size_ac,
                  B, ini_br + new_size_ac, ini_bc + new_size_bc,
                  C, ini_cr, ini_cc + new_size_bc,
                  new_size_ar, size_ac - new_size_ac, size_bc - new_size_bc,
                  min_size, 4*num_threads);
    pthread_create(&t6, NULL, &matmul_pt_rec, (void*)a_tmp);
    a_tmp = create_argument(A, ini_ar + new_size_ar, ini_ac + new_size_ac,
                  B, ini_br + new_size_ac, ini_bc,
                  C, ini_cr + new_size_ar, ini_cc,
                  size_ar - new_size_ar, size_ac - new_size_ac, new_size_bc,
                  min_size, 4*num_threads);
    pthread_create(&t7, NULL, &matmul_pt_rec, (void*)a_tmp);
    a_tmp = create_argument(A, ini_ar + new_size_ar, ini_ac + new_size_ac,
                  B, ini_br + new_size_ac, ini_bc + new_size_bc,
                  C, ini_cr + new_size_ar, ini_cc + new_size_bc,
                  size_ar - new_size_ar, size_ac - new_size_ac, size_bc - new_size_bc,
                  min_size, 4*num_threads);
    pthread_create(&t8, NULL, &matmul_pt_rec, (void*)a_tmp);

    pthread_join(t5, NULL);
    pthread_join(t6, NULL);
    pthread_join(t7, NULL);
    pthread_join(t8, NULL);

    free(a);
    return NULL;
}

void matmul_pt(Matrix A, Matrix B, Matrix C, uint64_t min_size) {
    Argument a = create_argument(A->matrix, 0, 0,
                                 B->matrix, 0, 0,
                                 C->matrix, 0, 0,
                                 A->n, A->m, B->m, min_size, 1);
    matmul_pt_rec((void*)a);
    printf("\n");
}
