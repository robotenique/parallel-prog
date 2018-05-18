#include "macros.h"
#include "openmpmatmul.h"
#include <immintrin.h>
#include <sys/sysinfo.h>
#include <sys/resource.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
// for (i = 0; i < a_n; i++) {
//      //printf("Thread #%d is doing row %d.\n",th_id,i);
//     for (j = 0; j < b_m; j++) {
//         double dot = 0;
//         for (k = 0; k < a_m; k++)
//             dot += a[i*a_m + k]*b[k*b_m + j];
//         c[i*c_m + j] = dot;
//     }
// }

void mmb_omp(double** A, int ini_ar, int ini_ac,
                    double** B, int ini_br, int ini_bc,
                    double** C, int ini_cr, int ini_cc,
                    int size_ar, int size_ac, int size_bc, int min_size) {
    if (!size_ac || !size_ar || !size_bc)
        return ;
    if (size_ar <= min_size && size_ac <= min_size && size_bc <= min_size) {
        int i, k, j;
        #pragma omp for private(i, j, k) schedule(static)
        for (i = 0; i < size_ar; i++) {
            for (k = 0; k < size_ac; k++) {
                for (j = 0; j < size_bc; j++)
                    C[ini_cr+i][ini_cc+j] += A[ini_ar+i][ini_ac+k]*B[ini_br+k][ini_bc+j];
            }
        }
        return ;
    }
    int new_size_ar = size_ar/2;
    int new_size_ac = size_ac/2;
    int new_size_bc = size_bc/2;
    #pragma omp task untied
    mmb_omp(A, ini_ar, ini_ac,
                   B, ini_br, ini_bc,
                   C, ini_cr, ini_cc,
                   new_size_ar, new_size_ac, new_size_bc, min_size);

    #pragma omp task untied
    mmb_omp(A, ini_ar, ini_ac,
                   B, ini_br, ini_bc + new_size_bc,
                   C, ini_cr, ini_cc + new_size_bc,
                   new_size_ar, new_size_ac, size_bc - new_size_bc, min_size);
    #pragma omp task untied
    mmb_omp(A, ini_ar + new_size_ar, ini_ac,
                   B, ini_br, ini_bc,
                   C, ini_cr + new_size_ar, ini_cc,
                   size_ar - new_size_ar, new_size_ac, new_size_bc, min_size);
    #pragma omp task untied
    mmb_omp(A, ini_ar + new_size_ar, ini_ac,
                   B, ini_br, ini_bc + new_size_bc,
                   C, ini_cr + new_size_ar, ini_cc + new_size_bc,
                   size_ar - new_size_ar, new_size_ac, size_bc - new_size_bc, min_size);

    #pragma omp task untied
    mmb_omp(A, ini_ar, ini_ac + new_size_ac,
                   B, ini_br + new_size_ac, ini_bc,
                   C, ini_cr, ini_cc,
                   new_size_ar, size_ac - new_size_ac, new_size_bc, min_size);
    #pragma omp task untied
    mmb_omp(A, ini_ar, ini_ac + new_size_ac,
                   B, ini_br + new_size_ac, ini_bc + new_size_bc,
                   C, ini_cr, ini_cc + new_size_bc,
                   new_size_ar, size_ac - new_size_ac, size_bc - new_size_bc, min_size);
    #pragma omp task untied
    mmb_omp(A, ini_ar + new_size_ar, ini_ac + new_size_ac,
                   B, ini_br + new_size_ac, ini_bc,
                   C, ini_cr + new_size_ar, ini_cc,
                   size_ar - new_size_ar, size_ac - new_size_ac, new_size_bc, min_size);
    #pragma omp task untied
    mmb_omp(A, ini_ar + new_size_ar, ini_ac + new_size_ac,
                   B, ini_br + new_size_ac, ini_bc + new_size_bc,
                   C, ini_cr + new_size_ar, ini_cc + new_size_bc,
                   size_ar - new_size_ar, size_ac - new_size_ac, size_bc - new_size_bc, min_size);
   #pragma omp taskwait
    return ;
}


void matmul_omp_2(Matrix A, Matrix B, Matrix C) {
    #pragma omp parallel
    #pragma omp master
    mmb_omp(A->matrix, 0, 0, B->matrix, 0, 0, C->matrix, 0, 0, A->n, A->m, B->m, 16);
}


void matmul_omp(MatrixArray A, MatrixArray B, MatrixArray C){
    u_int a_n = A->n;
    u_int a_m = A->m;
    u_int b_n = B->n;
    u_int b_m = B->m;
    u_int c_m = C->m;
    double* a = A->m_c;
    double* b = B->m_c;
    double* c = C->m_c;
    int th_id;
    omp_set_num_threads(get_nprocs());

    #pragma omp parallel
    {
        int i, j, k;
        th_id = omp_get_thread_num();
        #pragma omp for schedule(static)
        // for (i = 0; i < a_n; i++) {
        //      //printf("Thread #%d is doing row %d.\n",th_id,i);
        //     for (j = 0; j < b_m; j++) {
        //         double dot = 0;
        //         for (k = 0; k < a_m; k++)
        //             dot += a[i*a_m + k]*b[k*b_m + j];
        //         c[i*c_m + j] = dot;
        //     }
        // }
        for (i = 0; i < a_n; i++) {
             //printf("Thread #%d is doing row %d.\n",th_id,i);
             for (k = 0; k < b_n; k++) {
                double r = a[i*a_m + k];
                for (j = 0; j < b_m; j++) {
                    c[i*c_m + j] += r*b[k*b_m + j];
                }
             }
        }
    }
}
