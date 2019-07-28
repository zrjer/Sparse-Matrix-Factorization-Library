#include "cublas_demo.h"
#include "cublas_demo_kernel.cuh"

int main (int argc, char **argv)
{
    const int batch = BATCH, m = DIM_M, n = DIM_N, k = DIM_K;
    const double alpha = 1, beta = 0;
    size_t size_A, size_C;

    double *A, *h_A, *d_A;
    double *C, *h_C, *d_C;
    double **Aarray, **h_Aarray, **d_Aarray;
    double **Barray, **h_Barray, **d_Barray;
    double **Carray, **h_Carray, **d_Carray;
    double **Darray, **h_Darray, **d_Darray;
    struct syrk_meta *syrk_task, *h_syrk_task, *d_syrk_task;
    struct gemm_meta *gemm_task, *h_gemm_task, *d_gemm_task;

    cudaStream_t cudaStream;
    cublasHandle_t cublasHandle;

    struct timespec tp;
    double timestamp, timeSingle, timeBatch, timeKernel;

    FILE *file;

    size_A = k * ( n + m ) * sizeof(double);
    size_C = n * ( n + m ) * sizeof(double);

    cudaSetDevice (0);

    cudaStreamCreate ( &cudaStream );
    cublasCreate ( &cublasHandle );
    cublasSetStream ( cublasHandle, cudaStream );

    A = malloc ( batch * size_A );
    cudaMallocHost ( &(void*)h_A, batch * size_A );
    cudaMalloc ( &(void*)d_A, batch * size_A );

    C = malloc ( batch * size_C );
    cudaMallocHost ( &(void*)h_C, batch * size_C );
    cudaMalloc ( &(void*)d_C, batch * size_C );

    Aarray = malloc ( batch * sizeof ( double* ) );
    cudaMallocHost ( &(void*)h_Aarray, batch * sizeof ( double* ) );
    cudaMalloc ( &(void*)d_Aarray, batch * sizeof ( double* ) );

    Barray = malloc ( batch * sizeof ( double* ) );
    cudaMallocHost ( &(void*)h_Barray, batch * sizeof ( double* ) );
    cudaMalloc ( &(void*)d_Barray, batch * sizeof ( double* ) );

    Carray = malloc ( batch * sizeof ( double* ) );
    cudaMallocHost ( &(void*)h_Carray, batch * sizeof ( double* ) );
    cudaMalloc ( &(void*)d_Carray, batch * sizeof ( double* ) );

    Darray = malloc ( batch * sizeof ( double* ) );
    cudaMallocHost ( &(void*)h_Darray, batch * sizeof ( double* ) );
    cudaMalloc ( &(void*)d_Darray, batch * sizeof ( double* ) );

    for ( int idx = 0; idx < batch; idx++ )
    {
        Aarray[idx] = (void*)d_A + idx * size_A;
        Barray[idx] = (void*)d_A + idx * size_A + n * sizeof(double);
        Carray[idx] = (void*)d_C + idx * size_C;
        Darray[idx] = (void*)d_C + idx * size_C + n * sizeof(double);
    }

    memcpy ( h_Aarray, Aarray, batch * sizeof ( double* ) );
    cudaMemcpy ( d_Aarray, h_Aarray, batch * sizeof ( double* ), cudaMemcpyHostToDevice );

    memcpy ( h_Barray, Barray, batch * sizeof ( double* ) );
    cudaMemcpy ( d_Barray, h_Barray, batch * sizeof ( double* ), cudaMemcpyHostToDevice );

    memcpy ( h_Carray, Carray, batch * sizeof ( double* ) );
    cudaMemcpy ( d_Carray, h_Carray, batch * sizeof ( double* ), cudaMemcpyHostToDevice );

    memcpy ( h_Darray, Darray, batch * sizeof ( double* ) );
    cudaMemcpy ( d_Darray, h_Darray, batch * sizeof ( double* ), cudaMemcpyHostToDevice );

    syrk_task = malloc ( batch * sizeof ( struct syrk_meta ) );
    cudaMallocHost ( &(void*)h_syrk_task, batch * sizeof ( struct syrk_meta ) );
    cudaMalloc ( &(void*)d_syrk_task, batch * sizeof ( struct syrk_meta ) );

    gemm_task = malloc ( batch * sizeof ( struct gemm_meta ) );
    cudaMallocHost ( &(void*)h_gemm_task, batch * sizeof ( struct gemm_meta ) );
    cudaMalloc ( &(void*)d_gemm_task, batch * sizeof ( struct gemm_meta ) );

    for ( int idx = 0; idx < batch; idx++ )
    {
        for ( int j = 0; j < k; j++ )
            for ( int i = 0; i < n + m; i++ )
                A [ idx * k * ( n + m ) + j * ( n + m ) + i ] = j + i;

        syrk_task[idx].cublasHandle = cublasHandle;
        syrk_task[idx].n = n;
        syrk_task[idx].k = k;
        syrk_task[idx].alpha = alpha;
        syrk_task[idx].A = d_A + idx * k * ( n + m );
        syrk_task[idx].lda = n + m;
        syrk_task[idx].beta = beta;
        syrk_task[idx].C = d_C + idx * n * ( n + m );
        syrk_task[idx].ldc = n + m;

        gemm_task[idx].cublasHandle = cublasHandle;
        gemm_task[idx].m = m;
        gemm_task[idx].n = n;
        gemm_task[idx].k = k;
        gemm_task[idx].alpha = alpha;
        gemm_task[idx].A = d_A + idx * k * ( n + m ) + n;
        gemm_task[idx].lda = n + m;
        gemm_task[idx].B = d_A + idx * k * ( n + m );
        gemm_task[idx].ldb = n + m;
        gemm_task[idx].beta = beta;
        gemm_task[idx].C = d_C + idx * n * ( n + m ) + n;
        gemm_task[idx].ldc = n + m;
    }

    memcpy ( h_A, A, batch * size_A );
    cudaMemcpy ( d_A, h_A, batch * size_A, cudaMemcpyHostToDevice );

    memcpy ( h_syrk_task, syrk_task, batch * sizeof ( struct syrk_meta ) );
    cudaMemcpy ( d_syrk_task, h_syrk_task, batch * sizeof ( struct syrk_meta ), cudaMemcpyHostToDevice );

    memcpy ( h_gemm_task, gemm_task, batch * sizeof ( struct gemm_meta ) );
    cudaMemcpy ( d_gemm_task, h_gemm_task, batch * sizeof ( struct gemm_meta ), cudaMemcpyHostToDevice );

    {
        for ( int idx = 0; idx < batch; idx++ )
            for ( int j = 0; j < n; j++ )
                for ( int i = 0; i < n + m; i++ )
                    C [ idx * n * ( n + m ) + j * ( n + m ) + i ] = 1 / ( j + 1 ) + 1 / ( i + 1 );

        memcpy ( h_C, C, batch * size_C );
        cudaMemcpy ( d_C, h_C, batch * size_C, cudaMemcpyHostToDevice );

        clock_gettime ( CLOCK_REALTIME, &tp );
        timestamp = tp.tv_sec + ( double ) ( tp.tv_nsec ) / 1.0e9;

        for ( int idx = 0; idx < batch; idx++ )
        {
            cublasDsyrk ( syrk_task[idx].cublasHandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                    syrk_task[idx].n, syrk_task[idx].k,
                    &(syrk_task[idx].alpha), syrk_task[idx].A, syrk_task[idx].lda, &(syrk_task[idx].beta), syrk_task[idx].C, syrk_task[idx].ldc );
            cublasDgemm ( gemm_task[idx].cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                    gemm_task[idx].m, gemm_task[idx].n, gemm_task[idx].k,
                    &(gemm_task[idx].alpha), gemm_task[idx].A, gemm_task[idx].lda, gemm_task[idx].B, gemm_task[idx].ldb, &(gemm_task[idx].beta), gemm_task[idx].C, gemm_task[idx].ldc );
        }

        cudaStreamSynchronize ( cudaStream );

        clock_gettime ( CLOCK_REALTIME, &tp );
        timeSingle = tp.tv_sec + ( double ) ( tp.tv_nsec ) / 1.0e9 - timestamp;

        printf ("single: %lf sec\n", timeSingle);

        cudaMemcpy ( h_C, d_C, batch * size_C, cudaMemcpyDeviceToHost );
        memcpy ( C, h_C, batch * size_C );

        file = fopen ( "cublas_demo_single.log", "w" );

        for ( int idx = 0; idx < batch; idx++ )
        {
            for ( int j = 0; j < n; j++ )
            {
                for ( int i = 0; i < n + m; i++ )
                    fprintf ( file, "%le", ( j <= i ) ? C [ idx * n * ( n + m ) + j * ( n + m ) + i ] : 0 );
                fprintf ( file, "\n" );
            }
            fprintf ( file, "\n" );
        }

        fclose ( file );
    }

    {
        for ( int idx = 0; idx < batch; idx++ )
            for ( int j = 0; j < n; j++ )
                for ( int i = 0; i < n + m; i++ )
                    C [ idx * n * ( n + m ) + j * ( n + m ) + i ] = 1 / ( j + 1 ) + 1 / ( i + 1 );

        memcpy ( h_C, C, batch * size_C );
        cudaMemcpy ( d_C, h_C, batch * size_C, cudaMemcpyHostToDevice );

        clock_gettime ( CLOCK_REALTIME, &tp );
        timestamp = tp.tv_sec + ( double ) ( tp.tv_nsec ) / 1.0e9;

        cublasDgemmBatched ( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                n, n, k,
                &alpha, (const double**)d_Aarray, n + m, (const double**)d_Aarray, n + m, &beta, d_Carray, n + m, batch );
        cublasDgemmBatched ( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                m, n, k,
                &alpha, (const double**)d_Barray, n + m, (const double**)d_Aarray, n + m, &beta, d_Darray, n + m, batch );

        cudaStreamSynchronize ( cudaStream );

        clock_gettime ( CLOCK_REALTIME, &tp );
        timeBatch = tp.tv_sec + ( double ) ( tp.tv_nsec ) / 1.0e9 - timestamp;

        printf ("batch: %lf sec\n", timeBatch);

        cudaMemcpy ( h_C, d_C, batch * size_C, cudaMemcpyDeviceToHost );
        memcpy ( C, h_C, batch * size_C );

        file = fopen ( "cublas_demo_batch.log", "w" );

        for ( int idx = 0; idx < batch; idx++ )
        {
            for ( int j = 0; j < n; j++ )
            {
                for ( int i = 0; i < n + m; i++ )
                    fprintf ( file, "%le", ( j <= i ) ? C [ idx * n * ( n + m ) + j * ( n + m ) + i ] : 0 );
                fprintf ( file, "\n" );
            }
            fprintf ( file, "\n" );
        }

        fclose ( file );
    }

    {
        for ( int idx = 0; idx < batch; idx++ )
            for ( int j = 0; j < n; j++ )
                for ( int i = 0; i < n + m; i++ )
                    C [ idx * n * ( n + m ) + j * ( n + m ) + i ] = 1 / ( j + 1 ) + 1 / ( i + 1 );

        memcpy ( h_C, C, batch * size_C );
        cudaMemcpy ( d_C, h_C, batch * size_C, cudaMemcpyHostToDevice );

        clock_gettime ( CLOCK_REALTIME, &tp );
        timestamp = tp.tv_sec + ( double ) ( tp.tv_nsec ) / 1.0e9;

        launch_syrk_gemm ( batch, d_syrk_task, d_gemm_task, cudaStream );

        cudaStreamSynchronize ( cudaStream );

        clock_gettime ( CLOCK_REALTIME, &tp );
        timeKernel = tp.tv_sec + ( double ) ( tp.tv_nsec ) / 1.0e9 - timestamp;

        printf ("kernel: %lf sec\n", timeKernel);

        cudaMemcpy ( h_C, d_C, batch * size_C, cudaMemcpyDeviceToHost );
        memcpy ( C, h_C, batch * size_C );

        file = fopen ( "cublas_demo_kernel.log", "w" );

        for ( int idx = 0; idx < batch; idx++ )
        {
            for ( int j = 0; j < n; j++ )
            {
                for ( int i = 0; i < n + m; i++ )
                    fprintf ( file, "%le", ( j <= i ) ? C [ idx * n * ( n + m ) + j * ( n + m ) + i ] : 0 );
                fprintf ( file, "\n" );
            }
            fprintf ( file, "\n" );
        }

        fclose ( file );
    }

    return 0;
}
