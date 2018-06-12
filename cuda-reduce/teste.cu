#include <cstdio>
#include <iostream>
#define N 512
using namespace std;


__global__ void add(int *a, int *b, int *c) {
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}
void fill_array(int arr, int N, int val = 0) {
    for(int i = 0; i < N; arr[i++] = val);
}
int main(void){
    int a, b, c; // host copies of a, b, c
    int *d_a, *d_b, *d_c; // device copies of a, b, c
    int size = N*sizeof(int);

    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Allocate for host copies of a, b, c, and setup input values
    a = (int *)malloc(size); fill_array(a, N);
    b = (int *)malloc(size); fill_array(b, N, 1);
    c = (int *)malloc(size);



    // Setup input values
    a = 2;
    b = 7;

    // Copy input to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU
    add<<<N, 1>>>(d_a, d_b, d_c);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cout << "The value of C is: " << c << ".\n";
    return 0;
}