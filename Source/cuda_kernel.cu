#include "SparseFrame.h"

__global__ void mappedSubtract_kernel ( Float *d_A, Long nsrow, Float *d_C, Long nccol, Long ncrow, Long *d_RelativeMap )
{
    Long cj, ci;

    cj = blockIdx.x * blockDim.x + threadIdx.x;
    ci = blockIdx.y * blockDim.y + threadIdx.y;

    if ( cj < nccol && ci < ncrow )
        d_A [ d_RelativeMap[cj] * nsrow + d_RelativeMap[ci] ] -= d_C [ cj * ncrow + ci ];
}

void mappedSubtract ( Float *d_A, Long nsrow, Float *d_C, Long nccol, Long ncrow, Long *d_RelativeMap, cudaStream_t stream )
{
    dim3 block;
    dim3 thread(16, 16);

    block.x = ( nccol + thread.x - 1 ) / thread.x;
    block.y = ( ncrow + thread.y - 1 ) / thread.y;

    mappedSubtract_kernel <<< block, thread, 0, stream >>> ( d_A, nsrow, d_C, nccol, ncrow, d_RelativeMap );
}

__global__ void mappedSubtractComplex_kernel ( Complex *d_A, Long nsrow, Complex *d_C, Long nccol, Long ncrow, Long *d_RelativeMap )
{
    Long cj, ci;

    cj = blockIdx.x * blockDim.x + threadIdx.x;
    ci = blockIdx.y * blockDim.y + threadIdx.y;

    if ( cj < nccol && ci < ncrow )
    {
        d_A [ d_RelativeMap[cj] * nsrow + d_RelativeMap[ci] ].x -= d_C [ cj * ncrow + ci ].x;
        d_A [ d_RelativeMap[cj] * nsrow + d_RelativeMap[ci] ].y -= d_C [ cj * ncrow + ci ].y;
    }
}

void mappedSubtractComplex ( Complex *d_A, Long nsrow, Complex *d_C, Long nccol, Long ncrow, Long *d_RelativeMap, cudaStream_t stream )
{
    dim3 block;
    dim3 thread(16, 16);

    block.x = ( nccol + thread.x - 1 ) / thread.x;
    block.y = ( ncrow + thread.y - 1 ) / thread.y;

    mappedSubtractComplex_kernel <<< block, thread, 0, stream >>> ( d_A, nsrow, d_C, nccol, ncrow, d_RelativeMap );
}
