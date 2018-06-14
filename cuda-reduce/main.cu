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
                cache[cid] = minT(cache[cid + i + j], cache[cid + j]);
        }
        __syncthreads();
    }

    if (cid == 0) {
        for (int32_t i = 0; i < 9; i++)
            cache[blockIdx.x + i] = cache[i];
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
    cout << "=======REDUCED=======" << '\n';
    print_matrices(mat_reduced, 1);

    delete[] list_m;

    num_m = new_matrix_from_file(argv[1], &list_m);
    ecudaMalloc((void **)&d_list_m, num_m);
    ecudaMemcpy(list_m, d_list_m, 9*num_m*sizeof(int32_t), cudaMemcpyHostToDevice);

    reduce_min<<<num_m/NUM_THREADS, NUM_THREADS, 9*NUM_THREADS>>>(d_list_m, num_m);

    ecudaMemcpy(d_list_m, mat_reduced, 9*sizeof(int32_t), cudaMemcpyDeviceToHost);
    print_matrices(mat_reduced, 1);

    ecudaFree(d_list_m);
    delete[] list_m;
    delete[] mat_reduced;
    return 0;
}
