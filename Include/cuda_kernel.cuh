#ifndef INCLUDE_CUDA_KERNEL_CUH
#define INCLUDE_CUDA_KERNEL_CUH

#ifdef __cplusplus
extern "C"
{
#endif

    __global__ void mappedSubtract_kernel ( Float *d_A, Long lda, Float *d_C, Long nccol, Long ncrow, Long ldc, Long *d_RelativeMap );

    __host__ void mappedSubtract ( Float *d_A, Long lda, Float *d_C, Long nccol, Long ncrow, Long ldc, Long *d_RelativeMap, cudaStream_t stream );

    __global__ void mappedSubtractComplex_kernel ( Complex *d_A, Long lda, Complex *d_C, Long nccol, Long ncrow, Long ldc, Long *d_RelativeMap );

    __host__ void mappedSubtractComplex ( Complex *d_A, Long lda, Complex *d_C, Long nccol, Long ncrow, Long ldc, Long *d_RelativeMap, cudaStream_t stream );

#ifdef __cplusplus
}
#endif

#endif
