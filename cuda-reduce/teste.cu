#include <cstdio>
#include <iostream>
#include "error.h"
#include "cudaerror.h"
using namespace std;
const int N = 16384; // Num blocks
const int M = 128; // Num threads per block

__global__ void add(int *a, int *b, int *c) {
    int index = blockIdx.x*M + threadIdx.x;
    if (index < N)
        c[index] = a[index] + b[index];
}

void fill_array(int *arr, int N, int val) {
    for(int i = 0; i < N; i++)
        arr[i] = val;
}

void print_arr(int *arr, int N){
    for(int i = 0; i < N; i++){
        cout << arr[i] << " ";
        if(i % 45 == 44)
            cout << endl;
    }
    cout << endl;
}

int main(void){
    int *a, *b, *c; // host copies of a, b, c
    int *d_a, *d_b, *d_c; // device copies of a, b, c
    int size = N*sizeof(int);

    // Allocate space for device copies of a, b, c
    ecudaMalloc((void **)&d_a, size);
    ecudaMalloc((void **)&d_b, size);
    ecudaMalloc((void **)&d_c, size);

    // Allocate for host copies of a, b, c, and setup input values
    a = (int *)emalloc(size); fill_array(a, N, 20);
    b = (int *)emalloc(size); fill_array(b, N, -15);
    c = (int *)emalloc(size);

    // Copy input to device
    ecudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    ecudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU
    add<<<N/M, M>>>(d_a, d_b, d_c);

    // Copy result back to host
    ecudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    print_arr(c, N);
    // Cleanup
    free(a); free(b); free(c);
    ecudaFree(d_a); ecudaFree(d_b); ecudaFree(d_c);

    return 0;
}
