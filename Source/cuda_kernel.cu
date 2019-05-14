#include "SparseFrame.h"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
        (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val +
                    __longlong_as_double(assumed)));

    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

__global__ void createRelativeMap_kernel ( Long *d_RelativeMap, Long *d_Map, Long *d_Lsi, Long di_offset, Long ldd )
{
    Long di;

    di = blockIdx.x * blockDim.x + threadIdx.x;

    if ( di < ldd )
        d_RelativeMap[di] = d_Map [ d_Lsi [ di_offset + di ] ];
}

void createRelativeMap ( Long *d_RelativeMap, Long *d_Map, Long *d_Lsi, Long di_offset, Long ldd, cudaStream_t stream )
{
    dim3 block;
    dim3 thread(256);

    block.x = ( ldd + thread.x - 1 ) / thread.x;

    createRelativeMap_kernel <<< block, thread, 0, stream >>> ( d_RelativeMap, d_Map, d_Lsi, di_offset, ldd );
}

__global__ void mappedSubtract_kernel ( Float *d_A, Long lda, Float *d_C, Long nccol, Long ncrow, Long ldc, Long *d_RelativeMap )
{
    Long cj, ci;

    cj = blockIdx.x * blockDim.x + threadIdx.x;
    ci = blockIdx.y * blockDim.y + threadIdx.y;

    if ( cj < nccol && ci < ncrow )
        d_A [ d_RelativeMap[cj] * lda + d_RelativeMap[ci] ] -= d_C [ cj * ldc + ci ];
}

void mappedSubtract ( Float *d_A, Long lda, Float *d_C, Long nccol, Long ncrow, Long ldc, Long *d_RelativeMap, cudaStream_t stream )
{
    dim3 block;
    dim3 thread(16, 16);

    block.x = ( nccol + thread.x - 1 ) / thread.x;
    block.y = ( ncrow + thread.y - 1 ) / thread.y;

    mappedSubtract_kernel <<< block, thread, 0, stream >>> ( d_A, lda, d_C, nccol, ncrow, ldc, d_RelativeMap );
}

__global__ void mappedSubtractAtomic_kernel ( Float *d_A, Long lda, Float *d_C, Long nccol, Long ncrow, Long ldc, Long *d_RelativeMap )
{
    Long cj, ci;

    cj = blockIdx.x * blockDim.x + threadIdx.x;
    ci = blockIdx.y * blockDim.y + threadIdx.y;

    if ( cj < nccol && ci < ncrow )
        atomicAdd ( & ( d_A [ d_RelativeMap[cj] * lda + d_RelativeMap[ci] ] ), - d_C [ cj * ldc + ci ] );
}

void mappedSubtractAtomic ( Float *d_A, Long lda, Float *d_C, Long nccol, Long ncrow, Long ldc, Long *d_RelativeMap, cudaStream_t stream )
{
    dim3 block;
    dim3 thread(16, 16);

    block.x = ( nccol + thread.x - 1 ) / thread.x;
    block.y = ( ncrow + thread.y - 1 ) / thread.y;

    mappedSubtractAtomic_kernel <<< block, thread, 0, stream >>> ( d_A, lda, d_C, nccol, ncrow, ldc, d_RelativeMap );
}

__global__ void mappedSubtractComplex_kernel ( Complex *d_A, Long lda, Complex *d_C, Long nccol, Long ncrow, Long ldc, Long *d_RelativeMap )
{
    Long cj, ci;

    cj = blockIdx.x * blockDim.x + threadIdx.x;
    ci = blockIdx.y * blockDim.y + threadIdx.y;

    if ( cj < nccol && ci < ncrow )
    {
        d_A [ d_RelativeMap[cj] * lda + d_RelativeMap[ci] ].x -= d_C [ cj * ldc + ci ].x;
        d_A [ d_RelativeMap[cj] * lda + d_RelativeMap[ci] ].y -= d_C [ cj * ldc + ci ].y;
    }
}

void mappedSubtractComplex ( Complex *d_A, Long lda, Complex *d_C, Long nccol, Long ncrow, Long ldc, Long *d_RelativeMap, cudaStream_t stream )
{
    dim3 block;
    dim3 thread(16, 16);

    block.x = ( nccol + thread.x - 1 ) / thread.x;
    block.y = ( ncrow + thread.y - 1 ) / thread.y;

    mappedSubtractComplex_kernel <<< block, thread, 0, stream >>> ( d_A, lda, d_C, nccol, ncrow, ldc, d_RelativeMap );
}

__global__ void mappedSubtractComplexAtomic_kernel ( Complex *d_A, Long lda, Complex *d_C, Long nccol, Long ncrow, Long ldc, Long *d_RelativeMap )
{
    Long cj, ci;

    cj = blockIdx.x * blockDim.x + threadIdx.x;
    ci = blockIdx.y * blockDim.y + threadIdx.y;

    if ( cj < nccol && ci < ncrow )
    {
        atomicAdd ( & ( d_A [ d_RelativeMap[cj] * lda + d_RelativeMap[ci] ].x ), - d_C [ cj * ldc + ci ].x );
        atomicAdd ( & ( d_A [ d_RelativeMap[cj] * lda + d_RelativeMap[ci] ].y ), - d_C [ cj * ldc + ci ].y );
    }
}

void mappedSubtractComplexAtomic ( Complex *d_A, Long lda, Complex *d_C, Long nccol, Long ncrow, Long ldc, Long *d_RelativeMap, cudaStream_t stream )
{
    dim3 block;
    dim3 thread(16, 16);

    block.x = ( nccol + thread.x - 1 ) / thread.x;
    block.y = ( ncrow + thread.y - 1 ) / thread.y;

    mappedSubtractComplexAtomic_kernel <<< block, thread, 0, stream >>> ( d_A, lda, d_C, nccol, ncrow, ldc, d_RelativeMap );
}
