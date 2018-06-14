#include <cstdio>
#include <iostream>
#include "macros.h"
#include "error.h"
#include "cudaerror.h"
using namespace std;

__global__ void reduce_min( int32_t *mats, int32_t N ) {
    extern __shared__ int32_t cache[];
    int tid = 9*(threadIdx.x + blockIdx.x * blockDim.x);
    int cid = 9*threadIdx.x;

    for (int32_t i = 0; i < 9; i++)
        cache[cid + i] = mats[tid + i];

    __syncthreads();

    for (int32_t i = blockDim.x/2; i != 0; i >>= 1) {
        if (cid < i) {
            for (int32_t j = 0; j < 9; j++)
                cache[cid] = cache[cid + j] + ((cache[cid + i + j]-cache[cid + j])&((cache[cid + i + j]-cache[cid + j]) >> 31));
        }
        __syncthreads();
    }

    if (cid == 0) {
        for (int32_t i = 0; i < 9; i++)
            mats[blockIdx.x + i] = cache[i];
    }
}

int main(int argc, char const *argv[]) {
    set_prog_name("cuda-reduce");
    if(argc < 2)
        die("Wrong number of arguments!\nUsage ./main <path_matrices_file>");

    int32_t *list_m, *d_list_m, *mat_reduced;
    int32_t num_m = new_matrix_from_file(argv[1], &list_m);

    print_matrices(list_m, num_m);
    reduce_matrices_seq(list_m, num_m, &mat_reduced);
    cout << "=======REDUCED (SEQ)=======" << '\n';
    print_matrices(mat_reduced, 1);

    int32_t  *cuda_result = new int32_t[9*num_m];
    ecudaMalloc((void **)&d_list_m, 9*num_m*sizeof(int32_t));
    cout << "---- COPIA 1 ----\n";
    int32_t *coisa;
    new_matrix_from_file(argv[1], &coisa);
    ecudaMemcpy(d_list_m, coisa, 9*num_m*sizeof(int32_t), cudaMemcpyHostToDevice);
    cout << "---- COPIA 1 {end} ----\n";
    reduce_min<<<num_m/NUM_THREADS, NUM_THREADS, 9*NUM_THREADS>>>(d_list_m, num_m);
    cout << "---- COPIA 2 ----\n";
    ecudaMemcpy(cuda_result,d_list_m, 9*num_m*sizeof(int32_t), cudaMemcpyDeviceToHost);
    cout << "---- COPIA 2 {end} ----\n";
    print_matrices(cuda_result, 1);

    ecudaFree(d_list_m);
    delete[] list_m;
    delete[] mat_reduced;
    return 0;
}
