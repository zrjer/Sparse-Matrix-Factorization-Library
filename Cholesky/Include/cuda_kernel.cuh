#ifndef INCLUDE_CUDA_KERNEL_CUH
#define INCLUDE_CUDA_KERNEL_CUH

#ifdef __cplusplus
extern "C"
{
#endif

    __global__ void createMap_kernel ( Long *d_Map, Long *d_Lsi, Long sip_offset, Long lds );

    void createMap ( Long *d_Map, Long *d_Lsi, Long sip_offset, Long lds, cudaStream_t stream );

    __global__ void createRelativeMap_kernel ( Long *d_RelativeMap, Long *d_Map, Long *d_Lsi, Long dip_offset, Long ldd );

    __host__ void createRelativeMap ( Long *d_RelativeMap, Long *d_Map, Long *d_Lsi, Long dip_offset, Long ldd, cudaStream_t stream );

    __global__ void mappedSubtract_kernel ( int isAtomic, int isComplex, void *d_A, Long lda, void *d_C, Long cj_offset, Long ci_offset, Long nccol, Long ncrow, Long ldc, Long *d_RelativeMap );

    __host__ void mappedSubtract ( int isAtomic, int isComplex, void *d_A, Long lda, void *d_C, Long cj_offset, Long ci_offset, Long nccol, Long ncrow, Long ldc, Long *d_RelativeMap, cudaStream_t stream );

    __global__ void deviceSum_kernel ( int isComplex, void *d_A, void *d_B, void *d_C, Long nscol, Long nsrow, Long lda );

    __host__ void deviceSum ( int isComplex, void *d_A, void *d_B, void *d_C, Long nscol, Long nsrow, Long lda, cudaStream_t stream );

#ifdef __cplusplus
}
#endif

#endif
