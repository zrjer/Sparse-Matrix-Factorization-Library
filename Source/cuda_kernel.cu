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

__global__ void createRelativeMap_kernel ( Long *d_RelativeMap, Long di_offset, Long *d_Map, Long *d_Lsi, Long dip_offset, Long ldd )
{
    Long di;

    di = di_offset + blockIdx.x * blockDim.x + threadIdx.x;

    if ( di < ldd )
        d_RelativeMap[di] = d_Map [ d_Lsi [ dip_offset + di ] ];
}

void createRelativeMap ( Long *d_RelativeMap, Long di_offset, Long *d_Map, Long *d_Lsi, Long dip_offset, Long ldd, cudaStream_t stream )
{
    dim3 block, thread;

    thread.x = CUDA_BLOCKDIM_X * CUDA_BLOCKDIM_Y;
    block.x = ( ldd + thread.x - 1 ) / thread.x;

    createRelativeMap_kernel <<< block, thread, 0, stream >>> ( d_RelativeMap, di_offset, d_Map, d_Lsi, dip_offset, ldd );
}

__global__ void createRelativeMap_batched_kernel ( Long **d_RelativeMap, Long *di_offset, Long **d_Map, Long *d_Lsi, Long *dip_offset, Long *ldd )
{
    Long di;

    di = di_offset[blockIdx.x] + threadIdx.x;

    if ( di < ldd[blockIdx.x] )
        d_RelativeMap[blockIdx.x][di] = d_Map [blockIdx.x] [ d_Lsi [ dip_offset[blockIdx.x] + di ] ];
}

void createRelativeMap_batched ( Long batchSize, Long **d_RelativeMap, Long *di_offset, Long **d_Map, Long *d_Lsi, Long *dip_offset, Long *ldd, cudaStream_t stream )
{
    dim3 block, thread;

    thread.x = CUDA_BLOCKDIM_X * CUDA_BLOCKDIM_Y;
    block.x = batchSize;

    createRelativeMap_batched_kernel <<< block, thread, 0, stream >>> ( d_RelativeMap, di_offset, d_Map, d_Lsi, dip_offset, ldd );
}

__global__ void mappedSubtract_kernel ( int isAtomic, int isComplex, void *d_A, Long lda, void *d_C, Long cj_offset, Long ci_offset, Long nccol, Long ncrow, Long ldc, Long *d_RelativeMap )
{
    __shared__ Long shRelativeMap_j[CUDA_BLOCKDIM_X];
    __shared__ Long shRelativeMap_i[CUDA_BLOCKDIM_Y];

    Long cj, ci;

    cj = cj_offset + blockIdx.x * blockDim.x + threadIdx.x;
    ci = ci_offset + blockIdx.y * blockDim.y + threadIdx.y;

    if ( threadIdx.y == 0 )
        shRelativeMap_j[threadIdx.x] = d_RelativeMap[cj];

    if ( threadIdx.x == 0 )
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

__global__ void mappedSubtract_batched_kernel ( int isAtomic, int isComplex, void **d_A, Long *lda, void **d_C, Long *cj_offset, Long *ci_offset, Long *nccol, Long *ncrow, Long *ldc, Long **d_RelativeMap )
{
    Long cj, ci;

    cj = cj_offset[blockIdx.x] + threadIdx.x;
    ci = ci_offset[blockIdx.x] + threadIdx.y;

    if ( !isAtomic )
    {
        if ( !isComplex )
        {
            if ( cj < nccol[blockIdx.x] && ci < ncrow[blockIdx.x] )
                ( (Float*) ( d_A[blockIdx.x] ) ) [ d_RelativeMap[blockIdx.x][cj] * lda[blockIdx.x] + d_RelativeMap[blockIdx.x][ci] ] -= ( (Float*) ( d_C[blockIdx.x] ) )  [ cj * ldc[blockIdx.x] + ci ];
        }
        else
        {
            if ( cj < nccol[blockIdx.x] && ci < ncrow[blockIdx.x] )
            {
                ( (Complex*) ( d_A[blockIdx.x] ) ) [ d_RelativeMap[blockIdx.x][cj] * lda[blockIdx.x] + d_RelativeMap[blockIdx.x][ci] ].x -= ( (Complex*) ( d_C[blockIdx.x] ) ) [ cj * ldc[blockIdx.x] + ci ].x;
                ( (Complex*) ( d_A[blockIdx.x] ) ) [ d_RelativeMap[blockIdx.x][cj] * lda[blockIdx.x] + d_RelativeMap[blockIdx.x][ci] ].y -= ( (Complex*) ( d_C[blockIdx.x] ) ) [ cj * ldc[blockIdx.x] + ci ].y;
            }
        }
    }
    else
    {
        if ( !isComplex )
        {
            if ( cj < nccol[blockIdx.x] && ci < ncrow[blockIdx.x] )
                atomicAdd ( & ( ( (Float*) ( d_A[blockIdx.x] ) ) [ d_RelativeMap[blockIdx.x][cj] * lda[blockIdx.x] + d_RelativeMap[blockIdx.x][ci] ] ), - ( (Float*) ( d_C[blockIdx.x] ) ) [ cj * ldc[blockIdx.x] + ci ] );
        }
        else
        {
            if ( cj < nccol[blockIdx.x] && ci < ncrow[blockIdx.x] )
            {
                atomicAdd ( & ( ( (Complex*) ( d_A[blockIdx.x] ) ) [ d_RelativeMap[blockIdx.x][cj] * lda[blockIdx.x] + d_RelativeMap[blockIdx.x][ci] ].x ), - ( (Complex*) ( d_C[blockIdx.x] ) ) [ cj * ldc[blockIdx.x] + ci ].x );
                atomicAdd ( & ( ( (Complex*) ( d_A[blockIdx.x] ) ) [ d_RelativeMap[blockIdx.x][cj] * lda[blockIdx.x] + d_RelativeMap[blockIdx.x][ci] ].y ), - ( (Complex*) ( d_C[blockIdx.x] ) ) [ cj * ldc[blockIdx.x] + ci ].y );
            }
        }
    }
}

void mappedSubtract_batched ( Long batchSize, int isAtomic, int isComplex, void **d_A, Long *lda, void **d_C, Long *cj_offset, Long *ci_offset, Long *nccol, Long *ncrow, Long *ldc, Long **d_RelativeMap, cudaStream_t stream )
{
    dim3 block, thread;

    thread.x = CUDA_BLOCKDIM_X;
    thread.y = CUDA_BLOCKDIM_Y;
    block.x = batchSize;

    mappedSubtract_batched_kernel <<< block, thread, 0, stream >>> ( isAtomic, isComplex, d_A, lda, d_C, cj_offset, ci_offset, nccol, ncrow, ldc, d_RelativeMap );
}

__global__ void deviceSum_kernel ( int isComplex, void *d_A, void *d_A_, Long nscol, Long nsrow )
{
    Long sj, si;

    sj = blockIdx.x * blockDim.x + threadIdx.x;
    si = blockIdx.y * blockDim.y + threadIdx.y;

    if ( !isComplex )
    {
        if ( sj < nscol && si < nsrow )
            ( (Float*) d_A ) [ sj * nsrow + si ] += ( (Float*) d_A_ )  [ sj * nsrow + si ];
    }
    else
    {
        if ( sj < nscol && si < nsrow )
        {
            ( (Complex*) d_A ) [ sj * nsrow + si ].x += ( (Complex*) d_A_ ) [ sj * nsrow + si ].x;
            ( (Complex*) d_A ) [ sj * nsrow + si ].y += ( (Complex*) d_A_ ) [ sj * nsrow + si ].y;
        }
    }
}

void deviceSum ( int isComplex, void *d_A, void *d_A_, Long nscol, Long nsrow, cudaStream_t stream )
{
    dim3 block, thread;

    thread.x = CUDA_BLOCKDIM_X;
    thread.y = CUDA_BLOCKDIM_Y;
    block.x = ( nscol + thread.x - 1 ) / thread.x;
    block.y = ( nsrow + thread.y - 1 ) / thread.y;

    deviceSum_kernel <<< block, thread, 0, stream >>> ( isComplex, d_A, d_A_, nscol, nsrow );
}
