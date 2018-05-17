#include "macros.h"
#include "openmpmatmul.h"
#include <sys/sysinfo.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>


//
// void transpose(double *A, double *B, int n) {
//     int i,j;
//     for(i=0; i<n; i++) {
//         for(j=0; j<n; j++) {
//             B[j*n+i] = A[i*n+j];
//         }
//     }
// }
//
// void gemm(double *A, double *B, double *C, int n)
// {
//     int i, j, k;
//     for (i = 0; i < n; i++) {
//         for (j = 0; j < n; j++) {
//             double dot  = 0;
//             for (k = 0; k < n; k++) {
//                 dot += A[i*n+k]*B[k*n+j];
//             }
//             C[i*n+j ] = dot;
//         }
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
    int chunk = a_n;
    omp_set_num_threads(get_nprocs());

    #pragma omp parallel
    {
        int i, j, k;
        th_id = omp_get_thread_num();
        #pragma omp for schedule(guided, chunk)
        for (i = 0; i < a_n; i++) {
             printf("Thread #%d is doing row %d.\n",th_id,i); //Uncomment this line to see which thread is doing each row
            for (j = 0; j < b_m; j++) {
                double dot = 0;
                for (k = 0; k < a_m; k++)
                    dot += a[i*a_m + k]*b[k*b_m + j];
                c[i*c_m + j] = dot;
            }
        }
    }
}


// void gemm_omp(double *A, double *B, double *C, int n)
// {
//     #pragma omp parallel
//     {
//         int i, j, k;
//         #pragma omp for
//         for (i = 0; i < n; i++) {
//             for (j = 0; j < n; j++) {
//                 double dot  = 0;
//                 for (k = 0; k < n; k++) {
//                     dot += A[i*n+k]*B[k*n+j];
//                 }
//                 C[i*n+j ] = dot;
//             }
//         }
//
//     }
// }
//
// void gemmT(double *A, double *B, double *C, int n)
// {
//     int i, j, k;
//     double *B2;
//     B2 = (double*)malloc(sizeof(double)*n*n);
//     transpose(B,B2, n);
//     for (i = 0; i < n; i++) {
//         for (j = 0; j < n; j++) {
//             double dot  = 0;
//             for (k = 0; k < n; k++) {
//                 dot += A[i*n+k]*B2[j*n+k];
//             }
//             C[i*n+j ] = dot;
//         }
//     }
//     free(B2);
// }
//
// void gemmT_omp(double *A, double *B, double *C, int n)
// {
//     double *B2;
//     B2 = (double*)malloc(sizeof(double)*n*n);
//     transpose(B,B2, n);
//     #pragma omp parallel
//     {
//         int i, j, k;
//         #pragma omp for
//         for (i = 0; i < n; i++) {
//             for (j = 0; j < n; j++) {
//                 double dot  = 0;
//                 for (k = 0; k < n; k++) {
//                     dot += A[i*n+k]*B2[j*n+k];
//                 }
//                 C[i*n+j ] = dot;
//             }
//         }
//
//     }
//     free(B2);
// }

// int main() {
//     int i, n;
//     double *A, *B, *C, dtime;
//
//     n=512;
//     A = (double*)malloc(sizeof(double)*n*n);
//     B = (double*)malloc(sizeof(double)*n*n);
//     C = (double*)malloc(sizeof(double)*n*n);
//     for(i=0; i<n*n; i++) { A[i] = rand()/RAND_MAX; B[i] = rand()/RAND_MAX;}
//
//     dtime = omp_get_wtime();
//     gemm(A,B,C, n);
//     dtime = omp_get_wtime() - dtime;
//     printf("%f\n", dtime);
//
//     dtime = omp_get_wtime();
//     gemm_omp(A,B,C, n);
//     dtime = omp_get_wtime() - dtime;
//     printf("%f\n", dtime);
//
//     dtime = omp_get_wtime();
//     gemmT(A,B,C, n);
//     dtime = omp_get_wtime() - dtime;
//     printf("%f\n", dtime);
//
//     dtime = omp_get_wtime();
//     gemmT_omp(A,B,C, n);
//     dtime = omp_get_wtime() - dtime;
//     printf("%f\n", dtime);
//
//     return 0;
//
// }
