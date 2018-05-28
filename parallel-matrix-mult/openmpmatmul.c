#include "macros.h"
#include "openmpmatmul.h"
#include <immintrin.h>
#include <sys/sysinfo.h>
#include <sys/resource.h>
#include <omp.h>
#include <stdio.h>
#include <inttypes.h>
#include <stdlib.h>
// for (i = 0; i < a_n; i++) {
//      //printf("Thread #%lu is doing row %lu.\n",th_id,i);
//     for (j = 0; j < b_m; j++) {
//         double dot = 0;
//         for (k = 0; k < a_m; k++)
//             dot += a[i*a_m + k]*b[k*b_m + j];
//         c[i*c_m + j] = dot;
//     }
// }


void mmo_omp(MatrixArray A, MatrixArray B, MatrixArray C) {
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
    ini_i = 0;
    for (i = 0; i < r_ac; i++) {
        diff_i = ceilDiff(i, size_ac, r_ac);
        ini_k = 0;
        for (k = 0; k < r_ar; k++) {
            diff_k = ceilDiff(k, size_ar, r_ar);
            ini_j = 0;
            for (j = 0; j < r_bc; j++) {
                diff_j = ceilDiff(j, size_bc, r_bc);
                double* A_inner = pA + ini_k*size_ac + ini_i;
                double* B_inner = pB + ini_i*size_bc + ini_j;
                double* C_inner = pC + ini_k*size_bc + ini_j;
                double *pB_inner, *pC_inner;
                uint64_t size_ac_inner = diff_i;
                uint64_t size_ar_inner = diff_k;
                uint64_t size_bc_inner = diff_j;
                uint64_t or_size_ac = size_ac;
                uint64_t or_size_bc = size_bc;
                uint64_t i_inner, j_inner, k_inner;
                double r_inner; /* Tirar isso daqui!*/
                for (i_inner = 0; i_inner < size_ar_inner; i_inner++) {
                    for (k_inner = 0; k_inner < size_ac_inner; k_inner++) {
                        r_inner = *A_inner++;
                        pB_inner = B_inner + k_inner*or_size_bc;
                        pC_inner = C_inner + i_inner*or_size_bc;
                        for (j_inner = 0; j_inner < size_bc_inner; j_inner++)
                            *pC_inner++ += r_inner* *pB_inner++;
                    }
                    A_inner += or_size_ac - size_ac_inner;
                }

                ini_j += diff_j;
            }
            ini_k += diff_k;
        }

        ini_i += diff_i;
    }

}

void matmul_omp2(MatrixArray A, MatrixArray B, MatrixArray C){
    mmo_omp(A, B, C);
}

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
