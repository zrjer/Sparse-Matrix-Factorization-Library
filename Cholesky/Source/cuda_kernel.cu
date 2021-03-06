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

__global__ void createMap_kernel ( Long *d_Map, Long *d_Lsi, Long sip_offset, Long lds )
{
    Long si;

    si = blockIdx.x * blockDim.x + threadIdx.x;

    if ( si < lds )
        d_Map [ d_Lsi [ sip_offset + si ] ] = si;
}

void createMap ( Long *d_Map, Long *d_Lsi, Long sip_offset, Long lds, cudaStream_t stream )
{
    dim3 block, thread;

    thread.x = CUDA_BLOCKDIM_X * CUDA_BLOCKDIM_Y;
    block.x = ( lds + thread.x - 1 ) / thread.x;

    createMap_kernel <<< block, thread, 0, stream >>> ( d_Map, d_Lsi, sip_offset, lds );
}

__global__ void createRelativeMap_kernel ( Long *d_RelativeMap, Long *d_Map, Long *d_Lsi, Long dip_offset, Long ldd )
{
    Long di;

    di = blockIdx.x * blockDim.x + threadIdx.x;

    if ( di < ldd )
        d_RelativeMap[di] = d_Map [ d_Lsi [ dip_offset + di ] ];
}

void createRelativeMap ( Long *d_RelativeMap, Long *d_Map, Long *d_Lsi, Long dip_offset, Long ldd, cudaStream_t stream )
{
    dim3 block, thread;

    thread.x = CUDA_BLOCKDIM_X * CUDA_BLOCKDIM_Y;
    block.x = ( ldd + thread.x - 1 ) / thread.x;

    createRelativeMap_kernel <<< block, thread, 0, stream >>> ( d_RelativeMap, d_Map, d_Lsi, dip_offset, ldd );
}

__global__ void mappedSubtract_kernel ( int isAtomic, int isComplex, void *d_A, Long lda, void *d_C, Long cj_offset, Long ci_offset, Long nccol, Long ncrow, Long ldc, Long *d_RelativeMap )
{
    __shared__ Long shRelativeMap_j[CUDA_BLOCKDIM_X];
    __shared__ Long shRelativeMap_i[CUDA_BLOCKDIM_Y];

    Long cj, ci;

    cj = cj_offset + blockIdx.x * blockDim.x + threadIdx.x;
    ci = ci_offset + blockIdx.y * blockDim.y + threadIdx.y;

    if ( threadIdx.y == 0 && cj < nccol )
        shRelativeMap_j[threadIdx.x] = d_RelativeMap[cj];

    if ( threadIdx.x == 0 && ci < ncrow )
        shRelativeMap_i[threadIdx.y] = d_RelativeMap[ci];

    __syncthreads();

    if ( !isAtomic )
    {
        if ( !isComplex )
        {
            if ( cj < nccol && ci < ncrow )
                ( (Float*) d_A ) [ shRelativeMap_j[threadIdx.x] * lda + shRelativeMap_i[threadIdx.y] ] -= ( (Float*) d_C )  [ cj * ldc + ci ];
        }
        else
        {
            if ( cj < nccol && ci < ncrow )
            {
                ( (Complex*) d_A ) [ shRelativeMap_j[threadIdx.x] * lda + shRelativeMap_i[threadIdx.y] ].x -= ( (Complex*) d_C ) [ cj * ldc + ci ].x;
                ( (Complex*) d_A ) [ shRelativeMap_j[threadIdx.x] * lda + shRelativeMap_i[threadIdx.y] ].y -= ( (Complex*) d_C ) [ cj * ldc + ci ].y;
            }
        }
    }
    else
    {
        if ( !isComplex )
        {
            if ( cj < nccol && ci < ncrow )
                atomicAdd ( & ( ( (Float*) d_A ) [ shRelativeMap_j[threadIdx.x] * lda + shRelativeMap_i[threadIdx.y] ] ), - ( (Float*) d_C ) [ cj * ldc + ci ] );
        }
        else
        {
            if ( cj < nccol && ci < ncrow )
            {
                atomicAdd ( & ( ( (Complex*) d_A ) [ shRelativeMap_j[threadIdx.x] * lda + shRelativeMap_i[threadIdx.y] ].x ), - ( (Complex*) d_C ) [ cj * ldc + ci ].x );
                atomicAdd ( & ( ( (Complex*) d_A ) [ shRelativeMap_j[threadIdx.x] * lda + shRelativeMap_i[threadIdx.y] ].y ), - ( (Complex*) d_C ) [ cj * ldc + ci ].y );
            }
        }
    }
}

void mappedSubtract ( int isAtomic, int isComplex, void *d_A, Long lda, void *d_C, Long cj_offset, Long ci_offset, Long nccol, Long ncrow, Long ldc, Long *d_RelativeMap, cudaStream_t stream )
{
    dim3 block, thread;

    thread.x = CUDA_BLOCKDIM_X;
    thread.y = CUDA_BLOCKDIM_Y;
    block.x = ( nccol + thread.x - 1 ) / thread.x;
    block.y = ( ncrow + thread.y - 1 ) / thread.y;

    mappedSubtract_kernel <<< block, thread, 0, stream >>> ( isAtomic, isComplex, d_A, lda, d_C, cj_offset, ci_offset, nccol, ncrow, ldc, d_RelativeMap );
}

__global__ void deviceSum_kernel ( int isComplex, void *d_A, void *d_B, void *d_C, Long nscol, Long nsrow, Long lda )
{
    Long sj, si;

    sj = blockIdx.x * blockDim.x + threadIdx.x;
    si = blockIdx.y * blockDim.y + threadIdx.y;

    if ( !isComplex )
    {
        if ( sj < nscol && si < nsrow )
            ( (Float*) d_A ) [ sj * lda + si ] = ( (Float*) d_B )  [ sj * lda + si ] + ( (Float*) d_C )  [ sj * lda + si ];
    }
    else
    {
        if ( sj < nscol && si < nsrow )
        {
            ( (Complex*) d_A ) [ sj * lda + si ].x = ( (Complex*) d_B ) [ sj * lda + si ].x + ( (Complex*) d_C ) [ sj * lda + si ].x;
            ( (Complex*) d_A ) [ sj * lda + si ].y = ( (Complex*) d_B ) [ sj * lda + si ].y + ( (Complex*) d_C ) [ sj * lda + si ].y;
        }
    }
}

void deviceSum ( int isComplex, void *d_A, void *d_B, void *d_C, Long nscol, Long nsrow, Long lda, cudaStream_t stream )
{
    dim3 block, thread;

    thread.x = CUDA_BLOCKDIM_X;
    thread.y = CUDA_BLOCKDIM_Y;
    block.x = ( nscol + thread.x - 1 ) / thread.x;
    block.y = ( nsrow + thread.y - 1 ) / thread.y;

    deviceSum_kernel <<< block, thread, 0, stream >>> ( isComplex, d_A, d_B, d_C, nscol, nsrow, lda );
}
