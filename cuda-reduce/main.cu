#include <cstdio>
#include <iostream>
#include "macros.h"
#include "error.h"
#include "cudaerror.h"
using namespace std;

/*
 * Function: reduce_min
 * --------------------------------------------------------
 * CUDA kernel that performs the minimum reduction operation in a group of
 * 'N' 3x3 int32_t valued matrices. The reduction is done using a tree structure
 * by each thread of the block. Each block writes the result of its own reduction
 * in global memory.
 *
 * @args mats : int32_t* , a big array with N/9 matrices. The 'matrices' are
 *              stored as plain numbers in one big array.
 *      N : int32_t, the number of matrices in mats
 *
 * @return
 */
__global__ void reduce_min( int32_t *mats, int32_t N ) {
    extern __shared__ int32_t cache[];
    int tid = 9*(threadIdx.x + blockIdx.x * blockDim.x);
    int cid = 9*threadIdx.x;

    // Copy the block of matrices to local shared memory
    for (int32_t i = 0; i < 9; i++)
        cache[cid + i] = mats[tid + i];

    __syncthreads();
    // Do the reduction by using a tree structure, in the shared memory to maximize bandwidth
    for (int32_t i = blockDim.x/2; i != 0; i >>= 1) {
        if (threadIdx.x < i) {
            for (int32_t j = 0; j < 9; j++)
                cache[cid + j] = min(cache[cid + 9*i + j], cache[cid + j]);
        }
        __syncthreads(); // Threads need to be synchronized each iteration for correctness
    }
    /* Only the first thread of each block writes the final reduction of
       the block in the corresponding line of the global memory
    */
    if (cid == 0) {
        for (int32_t i = 0; i < 9; i++)
            mats[9*blockIdx.x + i] = cache[i];
    }
}

/*
 * Function: main
 * --------------------------------------------------------
 * Main function, read the arguments, then create the array with all the
 * matrices. It then reduces the matrices using a sequential implementation,
 * and then transfer the initial matrices to the GPU, calls the Kernel to reduce
 * the matrices, transfer back to the Host, and reduces the final values,
 * checking against the reduction done in the sequential implementation.
 *
 */
int main(int argc, char const *argv[]) {
    set_prog_name("cuda-reduce");
    if(argc < 2)
        die("Wrong number of arguments!\nUsage ./main <path_matrices_file>");

    int32_t *list_m, *d_list_m, *reduced_seq, *reduced_cuda;
    int32_t num_m = new_matrix_from_file(argv[1], &list_m);

    reduce_matrices_seq(list_m, num_m, &reduced_seq);
    cout << "=======REDUCED (SEQ)=======" << '\n';
    print_matrices(reduced_seq, 1);

    int32_t num_blocks = num_m/NUM_THREADS;
    int32_t  *cuda_result = new int32_t[9*num_blocks]; // To store the reduction of each block
    ecudaMalloc((void **)&d_list_m, 9*num_m*sizeof(int32_t));

    ecudaMemcpy(d_list_m, list_m, 9*num_m*sizeof(int32_t), cudaMemcpyHostToDevice);

    reduce_min<<<num_blocks, NUM_THREADS, 9*NUM_THREADS*sizeof(int32_t)>>>(d_list_m, num_m);

    ecudaMemcpy(cuda_result, d_list_m, 9*num_blocks*sizeof(int32_t), cudaMemcpyDeviceToHost);
    /* cuda_result has the reduction of each block. We then need to reduce each
     * of these block's reduction to obtain the final reduction.*/
    reduce_matrices_seq(cuda_result, num_blocks, &reduced_cuda);

    cout << "=======REDUCED (CUDA)=======" << '\n';
    print_matrices(reduced_cuda, 1);

    if(is_equal(reduced_seq, reduced_cuda))
        cout << "SUCCESS!!\n";
    else
        cout << "YOU HAVE FAILED!!\n";

    ecudaFree(d_list_m);
    
    delete[] list_m;
    delete[] reduced_seq;
    delete[] cuda_result;
    delete[] reduced_cuda;
    return 0;
}
