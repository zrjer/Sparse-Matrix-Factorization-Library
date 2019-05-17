#ifndef INCLUDE_CUDA_KERNEL_CUH
#define INCLUDE_CUDA_KERNEL_CUH

#ifdef __cplusplus
extern "C"
{
#endif

    __global__ void createRelativeMap_kernel ( Long *d_RelativeMap, Long *d_Map, Long *d_Lsi, Long di_offset, Long ldd );

    __host__ void createRelativeMap ( Long *d_RelativeMap, Long *d_Map, Long *d_Lsi, Long di_offset, Long ldd, cudaStream_t stream );

    __global__ void createRelativeMap_batched_kernel ( Long **d_RelativeMap, Long **d_Map, Long *d_Lsi, Long *di_offset, Long *ldd );

    __host__ void createRelativeMap_batched ( Long batchSize, Long **d_RelativeMap, Long **d_Map, Long *d_Lsi, Long *di_offset, Long *ldd, cudaStream_t stream );

    __global__ void mappedSubtract_kernel ( int isAtomic, int isComplex, void *d_A, Long lda, void *d_C, Long nccol, Long ncrow, Long ldc, Long *d_RelativeMap );

    __host__ void mappedSubtract ( int isAtomic, int isComplex, void *d_A, Long lda, void *d_C, Long nccol, Long ncrow, Long ldc, Long *d_RelativeMap, cudaStream_t stream );

    __global__ void mappedSubtract_batched_kernel ( int isAtomic, int isComplex, void **d_A, Long *lda, void **d_C, Long *nccol, Long *ncrow, Long *ldc, Long **d_RelativeMap );

    __host__ void mappedSubtract_batched ( Long batchSize, int isAtomic, int isComplex, void **d_A, Long *lda, void **d_C, Long *nccol, Long *ncrow, Long *ldc, Long **d_RelativeMap, cudaStream_t stream );

#ifdef __cplusplus
}
#endif

#endif
