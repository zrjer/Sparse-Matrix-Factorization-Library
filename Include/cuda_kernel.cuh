#ifndef INCLUDE_CUDA_KERNEL_CUH
#define INCLUDE_CUDA_KERNEL_CUH

#ifdef __cplusplus
extern "C"
{
#endif

    __global__ void createRelativeMap_kernel ( Long *d_RelativeMap, Long *d_Map, Long *d_Lsi, Long di_offset, Long ldd );

    __host__ void createRelativeMap ( Long *d_RelativeMap, Long *d_Map, Long *d_Lsi, Long di_offset, Long ldd, cudaStream_t stream );

    __global__ void mappedSubtract_kernel ( Float *d_A, Long lda, Float *d_C, Long nccol, Long ncrow, Long ldc, Long *d_RelativeMap );

    __host__ void mappedSubtract ( Float *d_A, Long lda, Float *d_C, Long nccol, Long ncrow, Long ldc, Long *d_RelativeMap, cudaStream_t stream );

    __global__ void mappedSubtractAtomic_kernel ( Float *d_A, Long lda, Float *d_C, Long nccol, Long ncrow, Long ldc, Long *d_RelativeMap );

    __host__ void mappedSubtractAtomic ( Float *d_A, Long lda, Float *d_C, Long nccol, Long ncrow, Long ldc, Long *d_RelativeMap, cudaStream_t stream );

    __global__ void mappedSubtractComplex_kernel ( Complex *d_A, Long lda, Complex *d_C, Long nccol, Long ncrow, Long ldc, Long *d_RelativeMap );

    __host__ void mappedSubtractComplex ( Complex *d_A, Long lda, Complex *d_C, Long nccol, Long ncrow, Long ldc, Long *d_RelativeMap, cudaStream_t stream );

    __global__ void mappedSubtractComplexAtomic_kernel ( Complex *d_A, Long lda, Complex *d_C, Long nccol, Long ncrow, Long ldc, Long *d_RelativeMap );

    __host__ void mappedSubtractComplexAtomic ( Complex *d_A, Long lda, Complex *d_C, Long nccol, Long ncrow, Long ldc, Long *d_RelativeMap, cudaStream_t stream );

#ifdef __cplusplus
}
#endif

#endif
