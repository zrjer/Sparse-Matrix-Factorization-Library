#ifndef DEMO_CUBLAS_DEMO_H
#define DEMO_CUBLAS_DEMO_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

struct syrk_meta
{
    cublasHandle_t cublasHandle;
    int n;
    int k;
    double alpha;
    double *A;
    int lda;
    double beta;
    double *C;
    int ldc;
};

struct gemm_meta
{
    cublasHandle_t cublasHandle;
    int m;
    int n;
    int k;
    double alpha;
    double *A;
    int lda;
    double *B;
    int ldb;
    double beta;
    double *C;
    int ldc;
};

#endif
