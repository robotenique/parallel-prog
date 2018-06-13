/*
 * @author: João Gabriel Basi Nº USP: 9793801
 * @author: Juliano Garcia de Oliveira Nº USP: 9277086
 *
 * MAC0219
 *
 * Error functions library for CUDA.
 */

#include <stdlib.h>
#include <errno.h>
#include "error.h"
#include "cudaerror.h"

void ecudaMalloc(void** devptr, size_t size) {
    errno = 0;
    cudaError_t res = cudaMalloc(devptr, size);

    if (res != cudaSuccess) {
        print_error_msg("call to cudaMalloc failed: %s", cudaGetErrorString(res));
        exit(-1);
    }

    return ;
}

void ecudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind) {
    errno = 0;
    cudaError_t res = cudaMemcpy(dst, src, count, kind);

    if (res != cudaSuccess) {
        print_error_msg("call to cudaMemcpy failed: %s", cudaGetErrorString(res));
        exit(-1);
    }

    return ;
}

void ecudaFree(void* devptr) {
    errno = 0;
    cudaError_t res = cudaFree(devptr);

    if (res != cudaSuccess) {
        print_error_msg("call to cudaMemcpy failed: %s", cudaGetErrorString(res));
        exit(-1);
    }

    return ;
}
