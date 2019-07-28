#include "cublas_demo.h"
#include "cublas_demo_kernel.cuh"

__global__ void launch_syrk_kernel ( struct syrk_meta *d_syrk_task )
{
    __shared__ struct syrk_meta syrk_task;
    __shared__ double shA[DIM_K][DIM_N+PAD];

    int idx;

    int n, k, lda, ldc;
    double alpha, beta;
    double *A, *C;

    idx = blockIdx.x;

    if ( threadIdx.x == 0 && threadIdx.y == 0 )
        syrk_task = d_syrk_task[idx];

    __syncthreads();

    n = syrk_task.n;
    k = syrk_task.k;
    lda = syrk_task.lda;
    ldc = syrk_task.ldc;
    alpha = syrk_task.alpha;
    beta = syrk_task.beta;
    A = syrk_task.A;
    C = syrk_task.C;

    for ( int j = threadIdx.x; j < k; j += blockDim.x )
        for ( int i = threadIdx.y; i < n; i += blockDim.y )
            shA[j][i] = A [ j * lda + i ];

    __syncthreads();

    for ( int j = threadIdx.x; j < n; j += blockDim.x )
        for ( int i = threadIdx.y; i < n; i += blockDim.y )
        {
            double regC;
            regC = beta * C [ j * ldc + i ];
            for ( int kk = 0; kk < k; kk++ )
                regC += ( alpha * shA[kk][j] * shA[kk][i] );
            C[ j * ldc + i ] = regC;
        }
}

__global__ void launch_gemm_kernel ( struct gemm_meta *d_gemm_task )
{
    __shared__ struct gemm_meta gemm_task;
    __shared__ double shA[DIM_K][DIM_M+PAD], shB[DIM_K][DIM_N+PAD];

    int idx;

    int m, n, k, lda, ldb, ldc;
    double alpha, beta;
    double *A, *B, *C;

    idx = blockIdx.x;

    if ( threadIdx.x == 0 && threadIdx.y == 0 )
        gemm_task = d_gemm_task[idx];

    __syncthreads();

    m = gemm_task.m;
    n = gemm_task.n;
    k = gemm_task.k;
    lda = gemm_task.lda;
    ldb = gemm_task.ldb;
    ldc = gemm_task.ldc;
    alpha = gemm_task.alpha;
    beta = gemm_task.beta;
    A = gemm_task.A;
    B = gemm_task.B;
    C = gemm_task.C;

    for ( int j = threadIdx.y; j < k; j += blockDim.y )
        for ( int i = threadIdx.x; i < m; i += blockDim.x )
            shA[j][i] = A [ j * lda + i ];

    for ( int j = threadIdx.x; j < k; j += blockDim.x )
        for ( int i = threadIdx.y; i < n; i += blockDim.y )
            shB[j][i] = B [ j * ldb + i ];

    __syncthreads();

    for ( int j = threadIdx.x; j < n; j += blockDim.x )
        for ( int i = threadIdx.y; i < m; i += blockDim.y )
        {
            double regC;
            regC = beta * C [ j * ldc + i ];
            for ( int kk = 0; kk < k; kk++ )
                regC += ( alpha * shA [kk][i] * shB [kk][j] );
            C [ j * ldc + i ] = regC;
        }
}

void launch_syrk_gemm ( int batch, struct syrk_meta *d_syrk_task, struct gemm_meta *d_gemm_task, cudaStream_t stream )
{
    dim3 thread;

    thread.x = 16;
    thread.y = 64;

    launch_syrk_kernel <<<batch, thread, 0, stream>>> ( d_syrk_task );
    launch_gemm_kernel <<<batch, thread, 0, stream>>> ( d_gemm_task );
}
