#include <cstdio>
#include <iostream>
using namespace std;
const int N = 512;

__global__ void add(int *a, int *b, int *c) {
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

void fill_array(int *arr, int N, int val) {
    for(int i = 0; i < N; i++)
        arr[i] = val;
}

void print_arr(int *arr, int N){
    for(int i = 0; i < N; i++){
        cout << arr[i] << "";
        if(i % 40 == 0)
            cout << endl;
    }
}

int main(void){
    int *a, *b, *c; // host copies of a, b, c
    int *d_a, *d_b, *d_c; // device copies of a, b, c
    int size = N*sizeof(int);

    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Allocate for host copies of a, b, c, and setup input values
    a = (int *)malloc(size); fill_array(a, N, 0);
    b = (int *)malloc(size); fill_array(b, N, 1);
    c = (int *)malloc(size);

    // Copy input to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU
    add<<<N, 1>>>(d_a, d_b, d_c);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    print_arr(c, N);
    // Cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}