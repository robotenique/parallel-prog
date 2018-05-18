#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include "macros.h"
#include "ptmatmul.h"

void add(double** C, u_int ini_cr, u_int ini_cc,
         double** T, u_int ini_tr, u_int ini_tc,
         u_int size_r, u_int size_c) {
    if (!size_r || !size_c)
        return ;
    if (size_r == 1 && size_c == 1) {
        C[ini_cr][ini_cc] += T[ini_tr][ini_tc];
        return ;
    }
    int new_size_r = size_r/2;
    int new_size_c = size_c/2;
    add(C, ini_cr, ini_cc,
        T, ini_tr, ini_tc,
        new_size_r, new_size_c);
    add(C, ini_cr + new_size_r, ini_cc,
        T, ini_tr + new_size_r, ini_tc,
        size_r - new_size_r, new_size_c);
    add(C, ini_cr, ini_cc + new_size_c,
        T, ini_tr, ini_tc + new_size_c,
        new_size_r, size_c - new_size_c);
    add(C, ini_cr + new_size_r, ini_cc + new_size_c,
        T, ini_tr + new_size_r, ini_tc + new_size_c,
        size_r - new_size_r, size_c - new_size_c);
    return;
}

void* matmul_pt_rec(void* arg) {
    Argument a = (Argument)arg;
    if (!(a->size_ac) || !(a->size_ar) || !(a->size_bc)) {
        free(a);
        return NULL;
    }
    double** A = a->A;
    double** B = a->B;
    double** C = a->C;
    u_int size_ac = a->size_ac;
    u_int size_ar = a->size_ar;
    u_int size_bc = a->size_bc;
    u_int ini_ar = a->ini_ar;
    u_int ini_ac = a->ini_ac;
    u_int ini_br = a->ini_br;
    u_int ini_bc = a->ini_bc;
    u_int ini_cr = a->ini_cr;
    u_int ini_cc = a->ini_cc;
    u_int min_size = a->min_size;
    printf("%d, %d, %d, %d, %d\n", size_ac, size_ar, size_bc, ini_ac, ini_bc);
    if (size_ar <= min_size && size_ac <= min_size && size_bc <= min_size) {
        for (int i = 0; i < size_ar; i++) {
            for (int k = 0; k < size_ac; k++) {
                for (int j = 0; j < size_bc; j++)
                    C[ini_cr+i][ini_cc+j] += A[ini_ar+i][ini_ac+k]*B[ini_br+k][ini_bc+j];
            }
        }
        free(a);
        return NULL;
    }
    u_int new_size_ar = size_ar/2;
    u_int new_size_ac = size_ac/2;
    u_int new_size_bc = size_bc/2;
    pthread_t t1, t2, t3, t4, t5, t6, t7, t8;
    Argument a_tmp;

    Matrix T = new_matrix_clean(size_ar, size_bc);

    a_tmp = create_argument(A, ini_ar, ini_ac,
                  B, ini_br, ini_bc,
                  C, ini_cr, ini_cc,
                  new_size_ar, new_size_ac, new_size_bc, min_size);
    pthread_create(&t1, NULL, &matmul_pt_rec, (void*)a_tmp);
    a_tmp = create_argument(A, ini_ar, ini_ac,
                  B, ini_br, ini_bc + new_size_bc,
                  C, ini_cr, ini_cc + new_size_bc,
                  new_size_ar, new_size_ac, size_bc - new_size_bc, min_size);
    pthread_create(&t2, NULL, &matmul_pt_rec, (void*)a_tmp);
    a_tmp = create_argument(A, ini_ar + new_size_ar, ini_ac,
                  B, ini_br, ini_bc,
                  C, ini_cr + new_size_ar, ini_cc,
                  size_ar - new_size_ar, new_size_ac, new_size_bc, min_size);
    pthread_create(&t3, NULL, &matmul_pt_rec, (void*)a_tmp);
    a_tmp = create_argument(A, ini_ar + new_size_ar, ini_ac,
                  B, ini_br, ini_bc + new_size_bc,
                  C, ini_cr + new_size_ar, ini_cc + new_size_bc,
                  size_ar - new_size_ar, new_size_ac, size_bc - new_size_bc, min_size);
    pthread_create(&t4, NULL, &matmul_pt_rec, (void*)a_tmp);

    a_tmp = create_argument(A, ini_ar, ini_ac + new_size_ac,
                  B, ini_br + new_size_ac, ini_bc,
                  T->matrix, 0, 0,
                  new_size_ar, size_ac - new_size_ac, new_size_bc, min_size);
    pthread_create(&t5, NULL, &matmul_pt_rec, (void*)a_tmp);
    a_tmp = create_argument(A, ini_ar, ini_ac + new_size_ac,
                  B, ini_br + new_size_ac, ini_bc + new_size_bc,
                  T->matrix, 0, new_size_bc,
                  new_size_ar, size_ac - new_size_ac, size_bc - new_size_bc, min_size);
    pthread_create(&t6, NULL, &matmul_pt_rec, (void*)a_tmp);
    a_tmp = create_argument(A, ini_ar + new_size_ar, ini_ac + new_size_ac,
                  B, ini_br + new_size_ac, ini_bc,
                  T->matrix, new_size_ar, 0,
                  size_ar - new_size_ar, size_ac - new_size_ac, new_size_bc, min_size);
    pthread_create(&t7, NULL, &matmul_pt_rec, (void*)a_tmp);
    a_tmp = create_argument(A, ini_ar + new_size_ar, ini_ac + new_size_ac,
                  B, ini_br + new_size_ac, ini_bc + new_size_bc,
                  T->matrix, new_size_ar, new_size_bc,
                  size_ar - new_size_ar, size_ac - new_size_ac, size_bc - new_size_bc, min_size);
    pthread_create(&t8, NULL, &matmul_pt_rec, (void*)a_tmp);

    add(C, ini_cr, ini_cc, T->matrix, 0, 0, size_ar, size_bc);
    destroy_matrix(T);
    free(a);
    return NULL;
}

void matmul_pt(Matrix A, Matrix B, Matrix C, u_int min_size) {
    Argument a = create_argument(A->matrix, 0, 0, B->matrix, 0, 0, C->matrix, 0, 0, A->n, A->m, B->m, min_size);
    matmul_pt_rec((void*)a);
}
