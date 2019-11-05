#ifndef DEMO_CUBLAS_DEMO_KERNEL_CUH
#define DEMO_CUBLAS_DEMO_KERNEL_CUH

#include "cublas_demo.h"

#ifdef __cplusplus
extern "C" {
#endif

    __global__ void launch_syrk_gemm_kernel ( struct syrk_meta *d_syrk_task, struct gemm_meta *d_gemm_task );

    void launch_syrk_gemm ( int batch, struct syrk_meta *d_syrk_task, struct gemm_meta *d_gemm_task, cudaStream_t stream );

#ifdef __cplusplus
}
#endif

#endif
