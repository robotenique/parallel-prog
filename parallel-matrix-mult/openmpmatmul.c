#include "macros.h"
#include "openmpmatmul.h"
#include <sys/sysinfo.h>
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

void matmul_omp(MatrixArray A, MatrixArray B, MatrixArray C){
    u_int a_n = A->n;
    u_int a_m = A->m;
    u_int b_n = B->n;
    u_int b_m = B->m;
    u_int c_n = C->n;
    u_int c_m = C->m;
    double* a = A->m_c;
    double* b = B->m_c;
    double* c = C->m_c;
    int th_id;
    int chunk = a_n/get_nprocs();
    omp_set_num_threads(get_nprocs());

    #pragma omp parallel
    {
        int i, j, k;
        th_id = omp_get_thread_num();
        #pragma omp for schedule(guided, chunk)
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
