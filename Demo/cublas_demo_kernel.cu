#include "cublas_demo.h"
#include "cublas_demo_kernel.cuh"

__global__ void launch_syrk_gemm_kernel ( struct syrk_meta *d_syrk_task, struct gemm_meta *d_gemm_task )
{
    __shared__ double M[32][33], N[32][33];
    __shared__ struct syrk_meta syrk_task;
    __shared__ struct gemm_meta gemm_task;

    int idx;

    idx = blockIdx.x;

    if ( threadIdx.x == 0 && threadIdx.y == 0 )
    {
        syrk_task = d_syrk_task[idx];
        gemm_task = d_gemm_task[idx];
    }

    __syncthreads();

    {
        int n, k, lda, ldc;
        double alpha, beta;
        double *A, *C;

        n = syrk_task.n;
        k = syrk_task.k;
        lda = syrk_task.lda;
        ldc = syrk_task.ldc;
        alpha = syrk_task.alpha;
        beta = syrk_task.beta;
        A = syrk_task.A;
        C = syrk_task.C;

        for ( int j = threadIdx.x; j < n; j += blockDim.x )
            for ( int i = threadIdx.y; i < n; i += blockDim.y )
                M[j][i] = beta * C [ j * ldc + i ];

        for ( int j = threadIdx.x; j < n; j += blockDim.x )
            for ( int i = threadIdx.y; i < n; i += blockDim.y )
                for ( int kk = 0; kk < k; kk++ )
                    M[j][i] += ( alpha * A [ kk * lda + i ] * A [ kk * lda + j ] );

        for ( int j = threadIdx.x; j < n; j += blockDim.x )
            for ( int i = threadIdx.y; i < n; i += blockDim.y )
                C[ j * ldc + i ] = M[j][i];
    }

    {
        int m, n, k, lda, ldb, ldc;
        double alpha, beta;
        double *A, *B, *C;

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

        for ( int j = threadIdx.x; j < n; j += blockDim.x )
            for ( int i = threadIdx.y; i < m; i += blockDim.y )
                N[j][i] = beta * C [ j * ldc + i ];

        for ( int j = threadIdx.x; j < n; j += blockDim.x )
            for ( int i = threadIdx.y; i < m; i += blockDim.y )
                for ( int kk = 0; kk < k; kk++ )
                    N[j][i] += ( alpha * A [ kk * lda + i ] * B [ kk * ldb + j ] );

        for ( int j = threadIdx.x; j < n; j += blockDim.x )
            for ( int i = threadIdx.y; i < m; i += blockDim.y )
                C [ j * ldc + i ] = N[j][i];
    }
}

void launch_syrk_gemm ( int batch, struct syrk_meta *d_syrk_task, struct gemm_meta *d_gemm_task, cudaStream_t stream )
{
    dim3 thread;

    thread.x = 16;
    thread.y = 32;

    launch_syrk_gemm_kernel <<<batch, 1, 0, stream>>> ( d_syrk_task, d_gemm_task );
}
