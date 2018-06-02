/*
 * @author: João Gabriel Basi Nº USP: 9793801
 * @author: Juliano Garcia de Oliveira Nº USP: 9277086
 *
 * MAC0219
 *
 * Matrix multiplication using openMP
 */
#include "macros.h"
#include "openmpmatmul.h"
#include <immintrin.h>
#include <sys/sysinfo.h>
#include <sys/resource.h>
#include <omp.h>
#include <stdio.h>
#include <inttypes.h>
#include <stdlib.h>


void matmul_omp(MatrixArray A, MatrixArray B, MatrixArray C){
    uint64_t a_n = A->n;
    uint64_t a_m = A->m;
    uint64_t b_n = B->n;
    uint64_t b_m = B->m;
    uint64_t c_m = C->m;
    double* a = A->m_c;
    double* b = B->m_c;
    double* c = C->m_c;
    omp_set_num_threads(get_nprocs());

    #pragma omp parallel
    {
        uint64_t i, j, k;
        #pragma omp for schedule(static)
        for (i = 0; i < a_n; i++) {
             for (k = 0; k < b_n; k++) {
                double r = a[i*a_m + k];
                for (j = 0; j < b_m; j++) {
                    c[i*c_m + j] += r*b[k*b_m + j];
                }
             }
        }
    }
}
